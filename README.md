# OpenCV Deep Learning based Object Detection

This is a minimal web application built with Streamlit, OpenCV, and deep learning to perform object detection on images using a pre-trained DenseNet-121 model. The project allows users to upload an image file or provide a URL for image analysis.

## Demo

Check out the live demo: [OpenCV Object Detection Demo](https://tejjus-object-detector-opencv.streamlit.app/)

## Features

- Object detection using a pre-trained DenseNet-121 model.
- Supports image upload and URL input for analysis.
- Provides class labels and confidence scores for detected objects.

## How to Use

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/opencv-object-detection.git
    cd opencv-object-detection
    ```

2. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Application:**

    ```bash
    streamlit run app.py
    ```

4. **Open the App in your Browser:**

    Visit [http://localhost:8501](http://localhost:8501) in your web browser.

## Screenshots

![Screenshot from 2023-11-16 13-58-14](https://github.com/Tejjus/object-detector-opencv/assets/112795549/bf9bbcf2-f39a-4309-a218-b9005c075e17)
![Screenshot from 2023-11-16 13-57-57](https://github.com/Tejjus/object-detector-opencv/assets/112795549/a6feb242-956c-47f7-bd71-c195a2108ef1)
![Screenshot from 2023-11-16 13-58-52](https://github.com/Tejjus/object-detector-opencv/assets/112795549/ced5d083-c20f-4f4b-bac0-e082afa246cf)

## File Structure

- `object-detector-streamlit.py`: Main application script.
- `DenseNet_121.caffemodel`: Pre-trained DenseNet-121 model file.
- `DenseNet_121.prototxt`: Configuration file for the DenseNet-121 model.
- `classification_classes_ILSVRC2012.txt`: ImageNet class names file.

## Acknowledgments

- The DenseNet-121 model is used from [https://github.com/shicai/DenseNet-Caffe.
- ImageNet class names are sourced from [https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a].
