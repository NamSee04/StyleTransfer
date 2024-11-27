from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import torch
from PIL import Image
from io import BytesIO
from src.model import net
from src.model.loss import adaptive_instance_normalization, coral
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

@app.post("/style_transfer/")
async def style_transfer_endpoint(content: UploadFile = File(...), style: UploadFile = File(...), alpha: float = 1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_image = Image.open(BytesIO(await content.read())).convert("RGB")
    style_image = Image.open(BytesIO(await style.read())).convert("RGB")

    content_tf = test_transform(512, False)
    style_tf = test_transform(512, False)

    content_tensor = content_tf(content_image).unsqueeze(0).to(device)
    style_tensor = style_tf(style_image).unsqueeze(0).to(device)

    decoder = net.decoder
    vgg = net.vgg

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load('./src/ckpt/decoder.pth'))
    vgg.load_state_dict(torch.load('./src/ckpt/vgg_normalised.pth'))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    with torch.no_grad():
        output = style_transfer(vgg, decoder, content_tensor, style_tensor, alpha)

    output_image = transforms.ToPILImage()(output.squeeze().cpu())

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        output_image.save(tmp, format="JPEG")
        tmp_path = tmp.name

    return FileResponse(tmp_path, media_type="image/jpeg", filename="stylized_image.jpg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)