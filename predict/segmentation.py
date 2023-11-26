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
    model_urls = data['model_files']

    models_file_path = []
    for model_url in model_urls:
        # Define the local file path for the model

        local_model_path = os.path.basename(model_url)
        models_file_path.append(local_model_path)

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
    return models_file_path

models_file_paths = download_model()

def download_images(image_urls):
    images_folder = "images"
    image_paths = []  # List to store the file paths of downloaded images

    # Create the folder if it doesn't exist
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
        print(f"Created folder: {images_folder}")

    def check_image_type(url):
        if 'munich' in url:
            return 'test'
        elif 'zurich' in url:
            return 'train'
        elif 'lindau' in url:
            return 'val'
        else:
            return 'real'

    for idx, image_url in enumerate(image_urls):

        appx = check_image_type(image_url)
        image_name = f"image_{idx + 1}_{appx}.jpg"  # Naming convention for downloaded images

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
def load_model(model_file_path):
    # Load the state dictionary
    state_dict = torch.load(model_file_path,map_location=torch.device('cpu'))

    # Remove keys related to auxiliary classifiers from the state dictionary
    state_dict = {k: v for k, v in state_dict.items() if 'aux_classifier' not in k}

    model = models.segmentation.fcn_resnet101(weights=False, num_classes=35)
    model.load_state_dict(state_dict, strict=False)  # Set strict=False to skip missing keys
    
    return model

def predict(image, model_file_path='model_weights_epoch_1.pth'):
    model = load_model(model_file_path)
    
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
st.markdown(f"""# Road Traffic Segmentation

The project aims to segment images of road traffic, focusing on accurately delineating various elements within these images. PyTorch was the primary framework utilized, employing the fcn_resnet101 base model. Training was conducted on the cityscapes dataset, incorporating both gtFine and leftImg8bit datasets available at [Cityscapes Dataset](https://www.cityscapes-dataset.com/).

## Training Details
- **Model Used**: fcn_resnet101
- **Dataset**: Cityscapes (gtFine, leftImg8bit)
- **Hardware**: NVIDIA RTX 2080 with CUDA
- **Training Time**: Approximately 20 hours
- **Image Processing**:
  - Resized to 256x256 pixels
  - Color information preserved

The code repository for this project is available on [GitHub](https://github.com/swisscenturion/u-net-segmentation-of-streets-and-cars). The repository contains the codebase responsible for implementing the segmentation model and serves as a reference for further exploration or utilization.
""")

st.title('Image Segmentation')

# Function to load and display the selected image
def display_image(image_path):
    image = Image.open(image_path)
    st.image(image, caption=image_path, use_column_width=True)
    return image

def predict_button(model_epoche):
    if st.button('Predict', use_container_width=True):
        input_image_transformed, colored_mask = predict(image, model_epoche)
        col1, col2 = st.columns(2)
        col1.image(input_image_transformed, caption='Preprocessed Image', use_column_width=True)
        col1.write('The image is preprocessed by resizing it to 256x256 pixels and converting it to a tensor. The model is trained on images of this size.')

        col2.image(colored_mask, caption='Predicted Mask', use_column_width=True)
        col2.write('The model predicts a mask for the image. The mask is a 2D array with the same size as the image. Each pixel of the mask is assigned a class. The colors of the 32 possible classes are randomly assigned.')

col1, col2 = st.columns(2)
example_or_own = col1.selectbox('Do you want to upload your own image or use examples?',['Example', 'Own Image'])
col1.write('You can either upload your own image or use one of the examples. The examples are images from Munich, Zurich and Lindau, and have the same format as the images used for training the model. The Images named "real" are images from Zurich and Luzern from the real world and have a different format.')
model_epoche = col2.selectbox('Select the model epoche',models_file_paths)
col2.image({data['gif']}, caption="How the Epoche Performed")
col2.write('The model epoche defines the number of training epochs. The higher the number, the better the model is trained. ')
# print(models_file_paths)

if example_or_own == 'Example':
    st.subheader('Example Images')
    st.write('Select an image from the dropdown menu and click on "Predict" to see the segmentation mask.')
    # Create a dropdown to select an image
    selected_image = st.selectbox('Select an image', image_file_paths)

    if selected_image:
        image = display_image(selected_image)
        predict_button(model_epoche)

if example_or_own == "Own Image":
    st.subheader('Upload own images')
    st.write('Upload your own image and click on "Predict" to see the segmentation mask.')

    
    uploaded_file = st.file_uploader("Upload a File", accept_multiple_files=False, type=['png','jpg'])
    if uploaded_file:
        bytes_data = uploaded_file.read()
        image = Image.open(BytesIO(bytes_data))
        st.image(image, caption='Uploaded Image', use_column_width=True)
        predict_button(model_epoche)
    