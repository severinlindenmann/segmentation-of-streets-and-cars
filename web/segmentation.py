import streamlit as st
from PIL import Image
import yaml
import torch
import torch.nn as nn
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

# Define the SegNet model
#create different operations of the network opearations of the network
class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        # Define the layers here
        # Note: for conv, use a padding of (1,1) so that size is maintained
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3,padding = 1)
        self.bn = nn.BatchNorm2d(out_ch,momentum = 0.1)
        self.relu = nn.ReLU()
    def forward(self, x):
        # define forward operation using the layers above
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class down_layer(nn.Module):
    def __init__(self):
        super(down_layer, self).__init__()
        self.down = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) # use nn.MaxPool2d( )        
    def forward(self, x):
        x1,idx = self.down(x)
        return x1,idx

class un_pool(nn.Module):
    def __init__(self):
        super(un_pool, self).__init__()       
        self.un_pool = nn.MaxUnpool2d(kernel_size=2, stride=2) # use nn.Upsample() with mode bilinear
        
    
    def forward(self, x, idx,x1):
        #Take the indicies from maxpool layer
        x = self.un_pool(x,idx,output_size = x1.size())
        return x 

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        # 1 conv layer
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3,padding = 1)

    def forward(self, x):
        # Forward conv layer
        x = self.conv(x)
        return x

# use all above the individual operations to build the network 
class SegNet(nn.Module):
    def __init__(self, n_channels_in, n_classes):
        super(SegNet, self).__init__()
        self.conv1 = single_conv(n_channels_in,64)
        self.conv2 = single_conv(64,64)
        self.down1 = down_layer() # Maxpool with giving max indices to do unpooling later
        self.conv3 = single_conv(64,128)
        self.conv4 = single_conv(128,128)
        self.down2 = down_layer() # Maxpool with giving max indices to do unpooling later
        self.conv5 = single_conv(128,256)
        self.conv6 = single_conv(256,256)
        self.conv7 = single_conv(256,256)
        self.down3 = down_layer() # Maxpool with giving max indices to do unpooling later
        self.conv8 = single_conv(256,512)
        self.conv9 = single_conv(512,512)
        self.conv10 = single_conv(512,512)
        self.down4 = down_layer() # Maxpool with giving max indices to do unpooling later
        self.conv11 = single_conv(512,512)
        self.conv12 = single_conv(512,512)
        self.conv13 = single_conv(512,512)
        self.down5 = down_layer()
        self.up1 = un_pool()
        self.conv14 = single_conv(512,512)
        self.conv15 = single_conv(512,512)
        self.conv16 = single_conv(512,512)
        self.up2 = un_pool()
        self.conv17 = single_conv(512,512)
        self.conv18 = single_conv(512,512)
        self.conv19 = single_conv(512,256)
        self.up3 = un_pool()
        self.conv20 = single_conv(256,256)
        self.conv21 = single_conv(256,256)
        self.conv22 = single_conv(256,128)
        self.up4 = un_pool()
        self.conv23 = single_conv(128,128)
        self.conv24 = single_conv(128,64)
        self.up5 = un_pool()
        self.conv25 = single_conv(64,64)
        self.outconv1 = outconv(64,n_classes)

    def forward(self, x):
        # Define forward pass
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3,idx1 = self.down1(x2) # skip connection <-------------------------
        x4 = self.conv3(x3)#                                                |
        x5 = self.conv4(x4)#                                                |
        x6,idx2 = self.down2(x5)# skip connection <-------------------      |
        x7 = self.conv5(x6)#                                         |      |
        x8 = self.conv6(x7)#                                         |      |
        x9 = self.conv7(x8)#                                         |      |
        x10,idx3 = self.down3(x9)# skip connection <-----------      |      |
        x11 = self.conv8(x10)#                                |      |      |
        x12 = self.conv9(x11)#                                |      |      | 
        x13 = self.conv10(x12)#                               |      |      |
        x14,idx4 = self.down4(x13)# skip connection <---      |      |      |
        x15 = self.conv11(x14)#                        |      |      |      |
        x16 = self.conv12(x15)#                        |      |      |      |
        x17 = self.conv13(x16)#                        |      |      |      |
        x18,idx5 = self.down5(x17)#                    |      |      |      |
        x19 = self.up1(x18,idx5,x17)#                  |      |      |      |
        x20 = self.conv14(x19)#                        |      |      |      |
        x21 = self.conv15(x20)#                        |      |      |      |
        x22 = self.conv16(x21)#                        |      |      |      |
        x23 = self.up2(x22,idx4,x13)# skip connection <-      |      |      |
        x24 = self.conv17(x23)#                               |      |      |
        x25 = self.conv18(x24)#                               |      |      |
        x26 = self.conv19(x25)#                               |      |      |
        x27 = self.up3(x26,idx3,x9)# skip connection <---------      |      |
        x28 = self.conv20(x27)#                                      |      |
        x29 = self.conv21(x28)#                                      |      |
        x30 = self.conv22(x29)#                                      |      |
        x31 = self.up4(x30,idx2,x5)# skip connection <----------------      |                                
        x32 = self.conv23(x31)#                                             |
        x33 = self.conv24(x32)#                                             |
        x34 = self.up4(x33,idx1,x2)# skip connection <-----------------------
        x35 = self.conv25(x34)
        x = self.outconv1(x35)
        ## Go up back to original dimension
        return x    

# Load the YAML file
with open('config.yaml', 'r') as file:
    data = yaml.safe_load(file)

def download_model():
    # Accessing the model_file
    model_urls = data['model_files']

    models_file_paths_resnet = []
    models_file_paths_segnet = []

    for model_url in model_urls:
        # Define the local file path for the model
        local_model_path = os.path.basename(model_url)

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

        # Separate URLs containing 'segnet' and 'resnet' into different lists
        if 'segnet' in model_url.lower():
            models_file_paths_segnet.append(local_model_path)
        elif 'resnet' in model_url.lower():
            models_file_paths_resnet.append(local_model_path)

    return models_file_paths_resnet, models_file_paths_segnet

models_file_paths_resnet, models_file_paths_segnet = download_model()

# Define a dictionary to hold the model paths based on the selection
model_paths_dict = {'resnet': models_file_paths_resnet, 'segnet': models_file_paths_segnet}

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

# Define your data transformation 
data_transforms_256 = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images
    transforms.ToTensor(),  # Convert to tensor
])

# Define your data transformation 
data_transforms_96 = transforms.Compose([
    transforms.Resize((96, 96)),  # Resize images
    transforms.ToTensor(),  # Convert to tensor
])

@st.cache_resource
def load_model_resnet(model_file_path):
    # Load the state dictionary
    state_dict = torch.load(model_file_path,map_location=torch.device('cpu'))

    # Remove keys related to auxiliary classifiers from the state dictionary
    state_dict = {k: v for k, v in state_dict.items() if 'aux_classifier' not in k}

    model = models.segmentation.fcn_resnet101(weights=False, num_classes=35)
    model.load_state_dict(state_dict, strict=False)  # Set strict=False to skip missing keys
    
    return model

@st.cache_resource
def load_model_segnet(model_file_path):
    model = SegNet(3,20) #one additional class for pixel ignored
    model.load_state_dict(torch.load(model_file_path,map_location=torch.device('cpu')))
    
    return model

def predict_segnet(image, model_file_path='model_segnet_weights_epoch_1.pth'):
    model = load_model_segnet(model_file_path)
    input_tensor = data_transforms_96(image).unsqueeze(0)

    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
    _, predicted = torch.max(output.data, 1) 
    
    print(predicted.shape)
    print(predicted)
    predicted = predicted.squeeze().cpu().numpy() 

    # Map predicted class indices to random colors
    num_classes = 20  # Number of classes
    height, width = predicted.shape
    print(predicted.shape)
    color_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Generate random colors for each class
    np.random.seed(42)  # Set a seed for reproducibility
    random_colors = np.random.randint(0, 256, size=(num_classes, 3), dtype=np.uint8)

    for label in range(num_classes):
        color_image[predicted == label] = random_colors[label]

    input_image_transformed = np.transpose(input_tensor.squeeze(0).cpu().numpy(), (1, 2, 0))

    return input_image_transformed, color_image


def predict_resnet(image, model_file_path='model_resnet_weights_epoch_1.pth'):
    model = load_model_resnet(model_file_path)
    
    # Apply the necessary transformations
    input_tensor = data_transforms_256(image).unsqueeze(0)

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

The project aims to segment images of road traffic, focusing on accurately delineating various elements within these images. PyTorch was the primary framework utilized, employing the fcn_resnet101 base model and a custom segnet model. Training was conducted on the cityscapes dataset, incorporating both gtFine and leftImg8bit datasets available at [Cityscapes Dataset](https://www.cityscapes-dataset.com/).
""")

model_selection = st.selectbox(
    'What model do you want to check out?',
    ('resnet', 'segnet'))

if model_selection == 'resnet':
    st.markdown(f"""
    ## Training Details
    - **Model Used**: fcn_resnet101
    - **Dataset**: Cityscapes (gtFine, leftImg8bit)
    - **Hardware**: NVIDIA RTX 2080 with CUDA
    - **Training Time**: Approximately 20 hours
    - **Classes**: 35
    - **Image Processing**:
    - Resized to 256x256 pixels
    - Color information preserved

    The code repository for this project is available on [GitHub](https://github.com/swisscenturion/u-net-segmentation-of-streets-and-cars). The repository contains the codebase responsible for implementing the segmentation model and serves as a reference for further exploration or utilization.
    """)
elif model_selection == 'segnet':
    st.markdown(f"""
    ## Training Details
    - **Model Used**: custom segnet
    - **Dataset**: Cityscapes (gtFine, leftImg8bit)
    - **Hardware**: Google Colab A100 GPU
    - **Training Time**: Approximately 10 hours
    - **Classes**: 19
    - **Image Processing**:
    - Resized to 96x96 pixels
    - Color information preserved

    The code repository for this project is available on [GitHub](https://github.com/swisscenturion/u-net-segmentation-of-streets-and-cars). The repository contains the codebase responsible for implementing the segmentation model and serves as a reference for further exploration or utilization.
    """)
else:
    pass

# Function to load and display the selected image
def display_image(image_path):
    image = Image.open(image_path)
    st.image(image, caption=image_path, use_column_width=True)
    return image

def predict_button(model_epoche,selected_image, model_selection):
    if st.button('Predict', use_container_width=True):
        if model_selection == 'resnet':
            input_image_transformed, colored_mask = predict_resnet(selected_image, model_epoche)
        else:
            input_image_transformed, colored_mask = predict_segnet(selected_image, model_epoche)

        col1, col2 = st.columns(2)

        col1.image(input_image_transformed, caption='Preprocessed Image', use_column_width=True)
        if model_selection == 'resnet':
            col1.write('The image is preprocessed by resizing it to 256x256 pixels and converting it to a tensor. The model is trained on images of this size.')
        elif model_selection == 'segnet':
            col1.write('The image is preprocessed by resizing it to 96x96 pixels and converting it to a tensor. The model is trained on images of this size.')

        col2.image(colored_mask, caption='Predicted Mask', use_column_width=True)
        col2.write('The model predicts a mask for the image. The mask is a 2D array with the same size as the image. Each pixel of the mask is assigned a class. The colors of the 32 possible classes are randomly assigned.')

col1, col2 = st.columns(2)

example_or_own = col1.selectbox('Do you want to upload your own image or use examples?',['Example', 'Own Image'])
col1.write('You can either upload your own image or use one of the examples. The examples are images from Munich, Zurich and Lindau, and have the same format as the images used for training the model. The Images named "real" are images from Zurich and Luzern that were not used for training and have a different format.')

# Select the model epoch based on the model_selection
model_epoche = col2.selectbox('Select the model epoche', model_paths_dict.get(model_selection, []))

if example_or_own == 'Example':
    st.subheader('Example Images')
    st.write('Select an image from the dropdown menu and click on "Predict" to see the segmentation mask.')
    # Create a dropdown to select an image
    selected_image = st.selectbox('Select an image', image_file_paths)

    if selected_image:
        image = display_image(selected_image)
        predict_button(model_epoche, image, model_selection)

if example_or_own == "Own Image":
    st.subheader('Upload own images')
    st.write('Upload your own image and click on "Predict" to see the segmentation mask.')

    uploaded_file = st.file_uploader("Upload a File", accept_multiple_files=False, type=['png','jpg'])
    if uploaded_file:
        bytes_data = uploaded_file.read()
        image = Image.open(BytesIO(bytes_data))
        st.image(image, caption='Uploaded Image', use_column_width=True)
        predict_button(model_epoche, image, model_selection)