import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.set_page_config(layout="wide")
st.title("Style Transfer")

# Sidebar for file upload
st.sidebar.header("Upload Images")
content_file = st.sidebar.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_file = st.sidebar.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])
alpha = st.sidebar.slider("Stylization Strength (Alpha)", 0.0, 1.0, 1.0)

# Columns for displaying images
col1, col2, col3 = st.columns(3)

def resize_image(image, width, height):
    return image.resize((width, height), Image.LANCZOS)

if content_file is not None:
    content_bytes = content_file.read()
    content_image = Image.open(BytesIO(content_bytes))

if style_file is not None:
    style_bytes = style_file.read()
    style_image = Image.open(BytesIO(style_bytes))

with col1:
    if content_file is not None:
        st.header("Content Image")
        resized_content_image = resize_image(content_image, 200, 200)
        st.image(resized_content_image, width=200)

    if style_file is not None:
        st.header("Style Image")
        resized_style_image = resize_image(style_image, 200, 200)
        st.image(resized_style_image, width=200)

if st.sidebar.button("Stylize"):
    if content_file is not None and style_file is not None:
        # Create new BytesIO instances for each request
        files_model1 = {
            'content': (content_file.name, BytesIO(content_bytes), content_file.type),
            'style': (style_file.name, BytesIO(style_bytes), style_file.type),
        }
        files_model2 = {
            'content': (content_file.name, BytesIO(content_bytes), content_file.type),
            'style': (style_file.name, BytesIO(style_bytes), style_file.type),
        }

        # Request to first model
        response1 = requests.post(
            "http://localhost:8000/style_transfer/",
            files=files_model1,
            data={'alpha': alpha}
        )

        # Request to second model
        response2 = requests.post(
            "http://localhost:8000/style_transfer_model2/",
            files=files_model2
        )

        if response1.status_code == 200:
            image1 = Image.open(BytesIO(response1.content))
            col2.header("Stylized Image (StyleT Model)")
            col2.image(image1)
        else:
            st.error(f"Error in style transfer with Model 1: {response1.status_code} - {response1.text}")

        if response2.status_code == 200:
            image2 = Image.open(BytesIO(response2.content))
            col3.header("Stylized Image (AniGAN Model)")
            col3.image(image2)
        else:
            st.error(f"Error in style transfer with Model 2: {response2.status_code} - {response2.text}")
    else:
        st.error("Please upload both content and style images")