import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm
import wandb

from model.net import Net, vgg, decoder
from sampler import InfiniteSamplerWrapper

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count, args):
    """Adjust learning rate according to decay."""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content_dir', type=str, default='./data/face2anime/train/trainA',
                        help='Directory path to a batch of content images')
    parser.add_argument('--style_dir', type=str, default='./data/face2anime/train/trainB',
                        help='Directory path to a batch of style images')
    parser.add_argument('--vgg', type=str, default='./ckpt/vgg_normalised.pth')

    # Training options
    parser.add_argument('--save_dir', default='./experiments',
                        help='Directory to save the model')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--max_iter', type=int, default=17500)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--style_weight', type=float, default=30.0)
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--n_threads', type=int, default=8)
    parser.add_argument('--save_model_interval', type=int, default=1750)
    parser.add_argument('--wandb_project', type=str, default='style_transfer',
                        help='Wandb project name for logging')
    args = parser.parse_args()

    # Initialize Weights & Biases
    wandb.init(project=args.wandb_project, config=vars(args))
    wandb.run.name = f"style-transfer-{wandb.run.id}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    network = Net(vgg, decoder)
    network.train()
    network.to(device)

    content_tf = train_transform()
    style_tf = train_transform()

    content_dataset = FlatFolderDataset(args.content_dir, content_tf)
    style_dataset = FlatFolderDataset(args.style_dir, style_tf)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads))

    optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

    for i in tqdm(range(args.max_iter)):
        adjust_learning_rate(optimizer, iteration_count=i, args=args)
        content_images = next(content_iter).to(device)
        style_images = next(style_iter).to(device)
        loss_c, loss_s = network(content_images, style_images)
        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s
        loss = loss_c + loss_s

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log to wandb
        wandb.log({
            'iteration': i + 1,
            'loss_content': loss_c.item(),
            'loss_style': loss_s.item(),
            'loss_total': loss.item(),
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            state_dict = decoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, save_dir /
                       f'decoder_iter_{i + 1}.pth')
            wandb.save(str(save_dir / f'decoder_iter_{i + 1}.pth'))
