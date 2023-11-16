from requests.models import MissingSchema
import streamlit as st
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO

st.title("OpenCV Deep Learning based Object Detection")

@st.cache_resource
# Load the model and class names.
def load_model():
  # Read the ImageNet class names.
  with open('classification_classes_ILSVRC2012.txt', 'r') as f:
        image_net_names = f.read().split('\n')

  # Final class names, picking just the first name if multiple in the class.
  class_names = [name.split(',')[0] for name in image_net_names]

  # Load the neural network model.
  model = cv2.dnn.readNet(
        model='DenseNet_121.caffemodel',
        config='DenseNet_121.prototxt',
        framework='Caffe')
  return model, class_names

def classify(model, image, class_names):
    # Remove alpha channel if found.
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Create blob from image using values specified by the model:
    blob = cv2.dnn.blobFromImage(
        image=image, scalefactor=0.017, size=(224, 224), mean=(104, 117, 123))
    
    # Set the input blob for the neural network and pass through network.
    model.setInput(blob)
    outputs = model.forward()

    # Get the final output scores and make all the outputs 1D.
    final_outputs = outputs[0]

    # Make all the outputs 1D.
    final_outputs = final_outputs.reshape(1000, 1)
    label_id = np.argmax(final_outputs)

    # Convert the output scores to softmax probabilities.
    probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))
    final_prob = np.max(probs) * 100.

    # Map the max confidence to the class label names.
    out_name = class_names[label_id]
    out_text = f"Class: {out_name}, Confidence: {final_prob:.1f}%"

    return out_text

def header(text):
    st.markdown(
        '<p style="background-color:#0066cc;color:#33ff33;font-size:24px;'
        f'border-radius:2%;" align="center">{text}</p>',
        unsafe_allow_html=True)
    
net, class_names = load_model()

# Create the file uploader widget
img_file_buffer = st.file_uploader("Choose a file or Camera", type=['jpg', 'jpeg', 'png'])
st.text('OR')
url = st.text_input('Enter URL')

image = "empty"

if img_file_buffer is not None:
    # Read the image buffer
    image = np.array(Image.open(img_file_buffer))
    st.image(image, caption='Uploaded Image', use_column_width=True)
elif url != '':
    try:
        response = requests.get(url)
        image = np.array(Image.open(BytesIO(response.content)))
        st.image(image)
    except MissingSchema as err:
        st.header('Invalid URL, Try Again!')
        print(err)
    except UnidentifiedImageError as err:
        st.header('URL has no Image, Try Again!')
        print(err)

try:
    # Call the classifier function and get the class label and probability of the image.
    out_text = classify(net, image, class_names)
    st.markdown(f'<p style="font-size:18px;">{out_text}</p>', unsafe_allow_html=True)
except:
    st.markdown(f'<p style="font-size:18px;">No Image Found</p>', unsafe_allow_html=True)
