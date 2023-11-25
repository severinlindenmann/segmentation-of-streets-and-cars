import streamlit as st
from PIL import Image, ImageEnhance
import yaml
import torch
from torchvision import models
import numpy as np
from torchvision import transforms
import requests
from io import BytesIO
import os.path

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

# Load the YAML file
with open('config.yaml', 'r') as file:
    data = yaml.safe_load(file)

def download_model():
    # Accessing the model_file
    model_url = data['model_file']
    local_model_path = "model_file.pth"

    # Check if the file already exists
    if os.path.exists(local_model_path):
        print(f"Model file already exists at: {local_model_path}")
    else:
        # Send a GET request to download the model
        response = requests.get(model_url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Save the model file locally
            with open(local_model_path, 'wb') as file:
                file.write(response.content)

            # Display the local path of the downloaded model
            print(f"Model file saved at: {local_model_path}")
        else:
            print("Failed to download the model")
    
download_model()

def download_images(image_urls):
    images_folder = "images"
    image_paths = []  # List to store the file paths of downloaded images

    # Create the folder if it doesn't exist
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
        print(f"Created folder: {images_folder}")

    for idx, image_url in enumerate(image_urls):
        image_name = f"image_{idx + 1}.jpg"  # Naming convention for downloaded images

        # Define the local file path for the image
        local_image_path = os.path.join(images_folder, image_name)

        # Check if the file already exists
        if os.path.exists(local_image_path):
            print(f"Image {image_name} already exists at: {local_image_path}")
        else:
            # Send a GET request to download the image
            response = requests.get(image_url)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Save the image file locally
                with open(local_image_path, 'wb') as file:
                    file.write(response.content)

                # Display the local path of the downloaded image
                print(f"Image {image_name} saved at: {local_image_path}")

        # Append the file path to the list
        image_paths.append(local_image_path)

    return image_paths

# Accessing the list of image URLs
image_urls = data['images']
image_file_paths = download_images(image_urls)

# Define your data transformation (you might need to customize these)
data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images
    transforms.ToTensor(),  # Convert to tensor
])

@st.cache_resource
def load_model():
    # Load the state dictionary
    state_dict = torch.load('model_file.pth',map_location=torch.device('cpu'))

    # Remove keys related to auxiliary classifiers from the state dictionary
    state_dict = {k: v for k, v in state_dict.items() if 'aux_classifier' not in k}

    model = models.segmentation.fcn_resnet101(weights=False, num_classes=35)
    model.load_state_dict(state_dict, strict=False)  # Set strict=False to skip missing keys
    
    return model

def predict(image):
    model = load_model()
    
    # Apply the necessary transformations
    input_tensor = data_transforms(image).unsqueeze(0)

    # Perform prediction
    with torch.no_grad():
        outputs = model(input_tensor)['out']  # Correct 'inputs' to 'input_tensor'
        _, predicted = torch.max(outputs, 1)

    # Convert the predicted image tensor to a numpy array
    predicted_numpy = predicted.squeeze(0).cpu().numpy()

    # Define label colors for 35 classes (replace this with your color mappings)
    np.random.seed(42)
    label_colors_35_classes = np.random.randint(0, 256, size=(35, 3), dtype=np.uint8)

    # Convert the predicted segmentation mask to a colored mask
    colored_mask = label_colors_35_classes[predicted_numpy]

    # Convert the input tensor back to a numpy array for plotting
    input_image_transformed = np.transpose(input_tensor.squeeze(0).cpu().numpy(), (1, 2, 0))
    return input_image_transformed, colored_mask


# Streamlit app
st.title(
    "U Net Segmenetation of Streets and Cars"
)

# Function to load and display the selected image
def display_image(image_path):
    image = Image.open(image_path)
    st.image(image, caption='Selected Image', use_column_width=True)
    return image

def predict_button():
    if st.button('Predict'):
        input_image_transformed, colored_mask = predict(image)
        col1, col2 = st.columns(2)
        col1.image(input_image_transformed, caption='Preprocessed Image', use_column_width=True)
        col2.image(colored_mask, caption='Predicted Mask', use_column_width=True)
        
example_or_own = st.selectbox('Do you want to upload your own image or use examples?',['Example', 'Own Image'])
if example_or_own == 'Example':
    st.title('Example Images')
    # Create a dropdown to select an image
    selected_image = st.selectbox('Select an image', image_file_paths)

    if selected_image:
        image = display_image(selected_image)
        predict_button()

if example_or_own == "Own Image":
    st.title('Upload own images')
    
    uploaded_file = st.file_uploader("Upload a File", accept_multiple_files=False, type=['png','jpg'])
    if uploaded_file:
        bytes_data = uploaded_file.read()
        image = Image.open(BytesIO(bytes_data))
        st.image(image, caption='Uploaded Image', use_column_width=True)
        predict_button()
    