import streamlit as st
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import sys
from io import BytesIO
from os import path
import requests
from requests.models import MissingSchema

# Constants.
INPUT_WIDTH = 640             
INPUT_HEIGHT = 640            
SCORE_THRESHOLD = 0.5         # Class score threshold, accepts only if score is above the threshold.
NMS_THRESHOLD = 0.45          # Non-maximum suppression threshold, higher values result in duplicate boxes per object 
CONFIDENCE_THRESHOLD = 0.45   # Confidence threshold, high values filter out low confidence detections

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 2.5
THICKNESS = 4

# Colors
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
RED = (0,0,255)
WHITE = (255,255,255)

st.title("Object Detection using YOLOv5")

frame = st.file_uploader("Choose a file or Camera", type=['jpg', 'jpeg', 'png'])
st.text('OR')
url = st.text_input('Enter URL')

if frame is not None:
    # Read the image buffer
    image = np.array(Image.open(frame))
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

# pre process input image as per model requirements.
def pre_process(input_image, net):
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)

    # Sets the input to the network.
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers.
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)
    # print(outputs[0].shape)

    return outputs

# post process output of the model.
def post_process(input_image, outputs):
    # Lists to hold respective values while unwrapping.
    class_ids = []
    confidences = []
    boxes = []

    # Rows.
    rows = outputs[0].shape[1]

    image_height, image_width = input_image.shape[:2]

    # Resizing factor.
    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT

    # Iterate through 25200 detections.
    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]

        # Discard bad detections and continue.
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]

            # Get the index of max class score.
            class_id = np.argmax(classes_scores)

            #  Continue if the class score is above threshold.
            if (classes_scores[class_id] > SCORE_THRESHOLD):
                confidences.append(confidence)
                class_ids.append(class_id)

                cx, cy, w, h = row[0], row[1], row[2], row[3]

                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                box = np.array([left, top, width, height])
                boxes.append(box)

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 4*THICKNESS)
        label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
        draw_label(input_image, label, left, top)

    return input_image

# Draw label on the image.
def draw_label(input_image, label, left, top):
    """Draw text onto image at location."""
    
    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle. 
    cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED);
    # Display text inside the rectangle.
    cv2.putText(input_image, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)

# function to put efficiency information on the image.
def put_efficiency(input_img, net):
  t, _ = net.getPerfProfile()
  label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
  print(label)
  cv2.putText(input_img, label, (20, 80), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)

# caching model weights.
@st.cache_resource
def load_model():
    modelWeights = "yolov5/models/yolov5l.onnx"
    net = cv2.dnn.readNet(modelWeights)
    return net

# Read Class names.
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
  classes = f.read().rstrip('\n').split('\n')

# Load the network.
net = load_model()

# Process image.
detections = pre_process(image, net)
img = post_process(image.copy(), detections)

# Put efficiency information.
put_efficiency(img, net)

# Display output image.
st.image(img, caption='Output Image', use_column_width=True)
