import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from src.model import net
from src.model.loss import adaptive_instance_normalization, coral


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0, interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


def main():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content', type=str, default='/home/namsee/Desktop/SCHOOL/cs431/StyleTransfer/img_content/imgHQ11172.png',
                        help='File path to the content image')
    parser.add_argument('--style', type=str, default='/home/namsee/Desktop/SCHOOL/cs431/StyleTransfer/img_style/152030.jpg',
                        help='File path to the style image, or multiple style '
                             'images separated by commas for interpolation')
    parser.add_argument('--vgg', type=str, default='./src/ckpt/vgg_normalised.pth')
    parser.add_argument('--decoder', type=str, default='/home/namsee/Desktop/SCHOOL/cs431/StyleTransfer/src/ckpt/z20.pth')

    # Additional options
    parser.add_argument('--content_size', type=int, default=512,
                        help='New (minimum) size for the content image, '
                             'keeping the original size if set to 0')
    parser.add_argument('--style_size', type=int, default=512,
                        help='New (minimum) size for the style image, '
                             'keeping the original size if set to 0')
    parser.add_argument('--crop', action='store_true',
                        help='Do center crop to create squared image')
    parser.add_argument('--save_ext', default='.jpg',
                        help='The extension name of the output image')
    parser.add_argument('--output', type=str, default='output',
                        help='Directory to save the output image(s)')

    # Advanced options
    parser.add_argument('--preserve_color', action='store_true',
                        help='If specified, preserve color of the content image')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='The weight that controls the degree of '
                             'stylization. Should be between 0 and 1')
    parser.add_argument('--style_interpolation_weights', type=str, default='',
                        help='The weight for blending the style of multiple style images')

    args = parser.parse_args()

    do_interpolation = ',' in args.style
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    content_path = Path(args.content)
    style_paths = args.style.split(',')

    if do_interpolation:
        assert args.style_interpolation_weights, 'Please specify interpolation weights'
        weights = [int(i) for i in args.style_interpolation_weights.split(',')]
        interpolation_weights = [w / sum(weights) for w in weights]
    else:
        interpolation_weights = None

    decoder = net.decoder
    vgg = net.vgg

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(args.decoder))
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    content_tf = test_transform(args.content_size, args.crop)
    style_tf = test_transform(args.style_size, args.crop)

    content = content_tf(Image.open(str(content_path))).unsqueeze(0).to(device)

    if do_interpolation:  # Interpolation of multiple styles
        style = torch.stack([style_tf(Image.open(str(Path(p)))) for p in style_paths]).to(device)
        content = content.expand_as(style)
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style, args.alpha, interpolation_weights)
        output_name = output_dir / f'{content_path.stem}_interpolation{args.save_ext}'
        save_image(output.cpu(), str(output_name))
    else:  # Single style
        for style_path in style_paths:
            style = style_tf(Image.open(str(Path(style_path)))).unsqueeze(0).to(device)
            if args.preserve_color:
                style = coral(style, content)
            with torch.no_grad():
                output = style_transfer(vgg, decoder, content, style, args.alpha)
            output_name = output_dir / f'{content_path.stem}_stylized_{Path(style_path).stem}{args.save_ext}'
            save_image(output.cpu(), str(output_name))


if __name__ == '__main__':
    main()
