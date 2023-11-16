import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Image Segmentation | Tool",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://severin.io",
        "Report a bug": "https://severin.io",
        "About": "Image Segmentation",
    },
)

st.title(
    "U Net Segmenetation of Streets and Cars"
)

def preprocess_image(image, output_image_path):
    # Size of the trained modell
    new_size = (2048, 1024)

    # Calculate the aspect ratio of the image
    original_ratio = image.size[0] / image.size[1]
    new_ratio = new_size[0] / new_size[1]

    # Resize the image to fit within the new dimensions while preserving aspect ratio and crop if necessary
    if original_ratio >= new_ratio:
        temp_height = new_size[1]
        temp_width = round(new_size[1] * original_ratio)
    else:
        temp_width = new_size[0]
        temp_height = round(new_size[0] / original_ratio)
        
    temp_size = (temp_width, temp_height)
    resized_image = image.resize(temp_size, Image.ANTIALIAS)
    
    # Calculate the cropping area
    left = (resized_image.width - new_size[0]) / 2
    top = (resized_image.height - new_size[1]) / 2
    right = (resized_image.width + new_size[0]) / 2
    bottom = (resized_image.height + new_size[1]) / 2

    # Crop the image
    resized_image = resized_image.crop((left, top, right, bottom))

    # Convert the image to greyscale
    resized_image = resized_image.convert("L")

    # Resize image to 512 x 512
    resized_image = resized_image.resize((512, 512), Image.ANTIALIAS)

    # Save the result
    resized_image.save(output_image_path, format="PNG")

# Streamlit app
st.title("Image Processing App")

# File uploader for image
picture = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if picture:
    # Display the uploaded image
    st.image(picture, caption="Uploaded Image", use_column_width=True)

    # Perform the resizing when a button is clicked
    if st.button("Resize"):
        # Specify the output path
        output_image_path = "image_processed.png"
        desired_size = (2048, 1024)

        # Resize the image without converting to grayscale and save as PNG
        image = Image.open(picture)
        cropped_image_size = preprocess_image(image, output_image_path)

        # Display the result
        st.success(f"Greyscaling and resizing to 512 x 512 Pixels complete.")
        st.image(output_image_path, caption="Resulting Image", use_column_width=True)

