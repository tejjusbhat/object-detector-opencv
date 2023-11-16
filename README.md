# OpenCV Deep Learning based Object Detection

This is a minimal web application built with Streamlit, OpenCV and deep learning to perform object detection on images using a pre-trained DenseNet-121 model. The project allows users to upload an image file or provide a URL for image analysis.

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
![main](https://github.com/Tejjus/object-detector-opencv/assets/112795549/22854bcd-ba4d-4d3c-84b3-3335a05fd337)
![balloon](https://github.com/Tejjus/object-detector-opencv/assets/112795549/617a6968-d609-486e-8062-c3d7dd144f09)
![tiger](https://github.com/Tejjus/object-detector-opencv/assets/112795549/190c8bf5-2b77-470d-b349-69c4ac6399fd)

    
## File Structure

- `object-detector-streamlit.py`: Main application script.
- `DenseNet_121.caffemodel`: Pre-trained DenseNet-121 model file.
- `DenseNet_121.prototxt`: Configuration file for the DenseNet-121 model.
- `classification_classes_ILSVRC2012.txt`: ImageNet class names file.

## Acknowledgments

- The DenseNet-121 model is used from [https://github.com/shicai/DenseNet-Caffe.
- ImageNet class names are sourced from [https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a].
