from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import torch
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from StyleT.src.model import net as style_net
from StyleT.src.model.loss import adaptive_instance_normalization
from AniGan.src.trainer import Trainer as anigan_trainer
from AniGan.src.utils import get_config as anigan_get_config
from torchvision import transforms
import torch.nn as nn
import os
import tempfile

app = FastAPI()

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def style_transfer(vgg, decoder, content, style, alpha=0.6, interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(content.device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

def _denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    if x is None:
        raise ValueError("Input tensor is None")
    out = (x + 1) / 2
    return out.clamp_(0, 1)

@app.post("/style_transfer/")
async def style_transfer_endpoint(
    content: UploadFile = File(...),
    style: UploadFile = File(...),
    alpha: float = Form(0.6)
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        content_bytes = await content.read()
        style_bytes = await style.read()
        content_img = Image.open(BytesIO(content_bytes)).convert("RGB")
        style_img = Image.open(BytesIO(style_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")

    content_tf = test_transform(size=256, crop=True)
    style_tf = test_transform(size=256, crop=True)

    content_tensor = content_tf(content_img).unsqueeze(0).to(device)
    style_tensor = style_tf(style_img).unsqueeze(0).to(device)

    # Load models
    decoder = style_net.decoder
    vgg = style_net.vgg

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load('/home/namsee/Desktop/SCHOOL/cs431/demo/StyleTransfer/StyleT/src/ckpt/z20.pth', map_location=device))
    vgg.load_state_dict(torch.load('/home/namsee/Desktop/SCHOOL/cs431/demo/StyleTransfer/StyleT/src/ckpt/vgg_normalised.pth', map_location=device))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    # Style transfer as in test.py
    with torch.no_grad():
        content_features = vgg(content_tensor)
        style_features = vgg(style_tensor)
        t = adaptive_instance_normalization(content_features, style_features)
        t = alpha * t + (1 - alpha) * content_features
        output = decoder(t)

    # Prepare output image
    output = output.clamp(0, 1)
    output_image = transforms.ToPILImage()(output.squeeze(0).cpu())
    output_image = output_image.resize((512, 512), Image.LANCZOS)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        output_image.save(tmp, format="JPEG")
        tmp_path = tmp.name

    return FileResponse(
        tmp_path,
        media_type="image/jpeg",
        filename="stylized_image.jpg"
    )

@app.post("/style_transfer_model2/")
async def style_transfer_model2_endpoint(content: UploadFile = File(...), style: UploadFile = File(...)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        content_bytes = await content.read()
        style_bytes = await style.read()
        content_img = Image.open(BytesIO(content_bytes)).convert('RGB')
        style_img = Image.open(BytesIO(style_bytes)).convert('RGB')
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Transform images as in test.py
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    content_tensor = transform(content_img).unsqueeze(0).to(device)
    style_tensor = transform(style_img).unsqueeze(0).to(device)

    # Load model as in test.py
    config_file = 'AniGan/src/configs/try4_final_r1p2.yaml'
    config = anigan_get_config(config_file)
    trainer = anigan_trainer(config)
    trainer.to(device)

    ckpt_path = 'AniGan/src/checkpoints/try4_final_r1p2/pretrained_face2anime.pt'
    trainer.load_ckpt(ckpt_path, map_location=device)
    trainer.eval()  # Ensure the model is in evaluation mode

    # Run model as in test.py
    with torch.no_grad():
        generated_img = trainer.model.evaluate_reference(content_tensor, style_tensor, device)
        if generated_img is None:
            raise HTTPException(status_code=500, detail="Generated image tensor is None")

        # Denormalize the generated image
        generated_img = (generated_img + 1) / 2
        generated_img = generated_img.clamp(0, 1)

    # Convert to PIL image and resize if necessary
    output_image = transforms.ToPILImage()(generated_img.squeeze(0).cpu())
    output_image = output_image.resize((512, 512), Image.LANCZOS)

    # Save output image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        output_image.save(tmp, format="JPEG")
        tmp_path = tmp.name

    return FileResponse(
        tmp_path,
        media_type="image/jpeg",
        filename="stylized_image.jpg"
    )
