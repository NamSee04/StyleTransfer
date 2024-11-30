import os
import argparse

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from src.trainer import Trainer
from src.utils import get_config


def _denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    if x is None:
        raise ValueError("Input tensor is None")
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def main(source_img_path, reference_img_path, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_file = 'src/configs/try4_final_r1p2.yaml'
    config = get_config(config_file)
    trainer = Trainer(config)
    trainer.to(device)

    ckpt_path = 'src/checkpoints/try4_final_r1p2/pretrained_face2anime.pt'
    trainer.load_ckpt(ckpt_path, map_location=device)
    trainer.eval()

    # prepare input image
    transform_list = [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    transform = transforms.Compose(transform_list)
    source_img = Image.open(source_img_path).convert('RGB')
    reference_img = Image.open(reference_img_path).convert('RGB')
    content_tensor = transform(source_img).unsqueeze(0).to(device)
    reference_tensor = transform(reference_img).unsqueeze(0).to(device)

    # Debugging: Print shapes of input tensors
    print(f"Content tensor shape: {content_tensor.shape}")
    print(f"Reference tensor shape: {reference_tensor.shape}")

    # run the model
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        generated_img = trainer.model.evaluate_reference(content_tensor, reference_tensor, device)
        if generated_img is None:
            raise ValueError("Generated image tensor is None")
        # Debugging: Print shape of generated image tensor
        print(f"Generated image tensor shape: {generated_img.shape}")
        name_part, ext_part = os.path.splitext(os.path.basename(source_img_path))
        save_file_name = f"{name_part}_anigan{ext_part}"
        save_file_path = os.path.join(output_dir, save_file_name)
        save_image(_denorm(generated_img), save_file_path, nrow=1, padding=0)
        print(f"Result is saved to: {save_file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_img', type=str, required=True, help="Source image path")
    parser.add_argument('--reference_img', type=str, required=True, help='Reference image path')
    parser.add_argument('--output_dir', type=str, default='result_dir', help='Directory path to save the result image')
    opts = parser.parse_args()

    main(opts.source_img, opts.reference_img, opts.output_dir)