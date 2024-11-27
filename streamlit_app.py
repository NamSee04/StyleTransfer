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
col1, col2 = st.columns(2)

def resize_image(image, width, height):
    return image.resize((width, height), Image.LANCZOS)

with col1:
    if content_file is not None:
        content_bytes = content_file.read()
        content_image = Image.open(BytesIO(content_bytes))
        st.header("Content Image")
        resized_content_image = resize_image(content_image, 200, 200)
        st.image(resized_content_image, width=200)

    if style_file is not None:
        style_bytes = style_file.read()
        style_image = Image.open(BytesIO(style_bytes))
        st.header("Style Image")
        resized_style_image = resize_image(style_image, 200, 200)
        st.image(resized_style_image, width=200)

if st.sidebar.button("Stylize"):
    if content_file is not None and style_file is not None:
        files = {
            'content': (content_file.name, BytesIO(content_bytes), content_file.type),
            'style': (style_file.name, BytesIO(style_bytes), style_file.type),
        }
        response = requests.post("http://localhost:8000/style_transfer/", files=files, data={'alpha': alpha})

        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            col2.header("Stylized Image")
            #resized_image = resize_image(image, 200, 200)
            col2.image(image)
        else:
            st.error("Error in style transfer")
    else:
        st.error("Please upload both content and style images")