
# Sketch Recognition with Gradio and TensorFlow

## Introduction
This project demonstrates how to build and deploy a machine learning-powered sketch recognition app using **Gradio**, **TensorFlow**, and **Docker**. The app is designed to classify handwritten digits (0-9) based on the MNIST dataset. It provides an interactive interface that lets users draw numbers on a sketchpad and receive real-time predictions.

---

## Features
- **Interactive Interface:** Classify handwritten digits using a sketchpad.
- **Pre-trained Model Integration:** Uses a TensorFlow-trained model for accurate predictions.
- **Dockerized Deployment:** Easily deployable on local or cloud infrastructure.
- **Customizable:** Adaptable to other datasets and use cases.

---

## Prerequisites
- Python 3.7 or later
- Docker installed on your machine
- Basic knowledge of Python and Docker
- TensorFlow and Gradio libraries (specified in `requirements.txt`)

---

## Project Structure
- `app.py`: Main application code for Gradio.
- `requirements.txt`: Lists required Python dependencies.
- `Dockerfile`: Configuration for containerizing the app.
- `model/`: Directory containing the pre-trained model file (`sketch_recognition_numbers_model.h5`).

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/sketch-recognition.git
cd sketch-recognition
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application Locally
```bash
python app.py
```

### 4. Build and Run the Docker Image
```bash
docker build . -t gradio_app:latest
docker run -p 8080:8080 gradio_app:latest
```

---

## Writing the Gradio App

The app includes a `predict` function for classification and a Gradio interface for interactivity. Below is an example:

```python
import gradio as gr
import tensorflow as tf
import cv2

model = tf.keras.models.load_model("model/sketch_recognition_numbers_model.h5")

def predict(img):
    img = cv2.resize(img, (28, 28)).reshape(1, 28, 28, 1)
    preds = model.predict(img)[0]
    return {label: float(pred) for label, pred in zip(labels, preds)}

gr.Interface(
    fn=predict,
    inputs="sketchpad",
    outputs=gr.outputs.Label(num_top_classes=3)
).launch()
```

---

## Building the Docker Image

To containerize the application, use the provided `Dockerfile`:

```dockerfile
FROM python:3.7
WORKDIR /workspace
ADD . /workspace
RUN pip install -r requirements.txt
CMD ["python3", "/workspace/app.py"]
```

Build the image using:
```bash
docker build . -t gradio_app:latest
```

---

## Extending the Project

This app can be adapted for other use cases:
1. Replace the pre-trained model with your own TensorFlow model.
2. Update the `predict` function in `app.py` to handle your model's requirements.
3. Adjust input/output formats as needed.

---

## Known Issues / Limitations
- Currently supports only digit classification from the MNIST dataset.
- Intended for testing purposes and not production-ready.
- Requires adjustments for non-standard datasets.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- [Gradio](https://gradio.app/)
- [TensorFlow](https://www.tensorflow.org/)
- MNIST dataset from [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/).

