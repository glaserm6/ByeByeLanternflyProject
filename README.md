# Bye-Bye-Lanternfly 🪰

## Project Overview

Bye-Bye-Lanternfly is a computer vision project designed to detect and count adult spotted lanternflies (*Lycorma delicatula*) in uploaded photographs. The goal is to provide a simple, accessible tool that helps users estimate the number of visible adult lanternflies on trees and identify potential infestations.

The project uses **YOLOv8n (You Only Look Once)**, a pretrained object-detection model fine-tuned on an adult spotted lanternfly dataset. Model training was conducted using Google Colab's GPU runtime, while the user interface was developed in Python using Flask.

Users can upload a photograph through a mobile-friendly web interface and receive:
- The total number of detected adult spotted lanternflies.
- An annotated image showing detected insects and bounding boxes.
- Detection confidence information.
- A preliminary indication of potential infestation based on the number of detected insects.

## Model & Training

The model was trained using a machine-screened version of the Roboflow v7 adult spotted lanternfly dataset.

| Configuration | Details |
|---|---|
| Model architecture | YOLOv8n |
| Initial weights | Pretrained YOLOv8n |
| Training epochs | 100 |
| Image size | 640 × 640 pixels |
| Hardware | Google Colab NVIDIA Tesla T4 GPU |
| Training time | Approximately 1.23 hours |
| Training images | 1,827 |
| Validation images | 438 |
| Testing images | 276 |

## Testing & Results

The latest trained model was evaluated on 276 test images containing 765 annotated adult spotted lanternflies. Counting performance was evaluated using a confidence threshold of 0.20 and an image size of 640 pixels.

### Counting Performance

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | **0.4674** |
| Root Mean Squared Error (RMSE) | 3.1027 |
| Exact-count accuracy | **89.13%** |
| Exact predicted counts | 246 / 276 |
| Predictions within one insect | **95.29%** |
| Predictions within one insect | 263 / 276 |


### Understanding the Results

The results demonstrate strong counting performance on the test dataset. However, densely grouped insects, overlapping objects, unusual lighting, and partial visibility remain challenges. In particular, some images with high insect densities produce substantial undercounting.

These metrics measure counting performance rather than conventional object-detection precision or mAP (as mAP was used previously and does not help explain the counting performance). 

## Running the Interface (if you want to implement this for yourself)

### Requirements
- Python 3.10+
- Flask
- Ultralytics
- OpenCV
- PyTorch

### Instructions

1. Clone the repository and navigate to its directory.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the trained model is located at `models/best.pt`.
4. Start the Flask application:

   ```bash
   python app.py
   ```

5. Open `http://127.0.0.1:5000` in your browser.

Upload an image to view the detection results and estimated lanternfly count.

## Future Improvements

Future development will focus on improving performance in challenging images through additional training data, image preprocessing (including contrast enhancement), and augmentation techniques that better represent overlapping and densely grouped lanternflies.

The model will also be evaluated against its current baseline to determine whether these improvements produce measurable gains in counting accuracy.

## Thank you for checking out my project!!!
