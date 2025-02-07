# Action Detection Model

## Table of Contents
- [Overview](#overview)
- [Technologies and Libraries Used](#technologies-and-libraries-used)
- [How to Run](#how-to-run)
- [Detailed Code Explanation](#detailed-code-explanation)
  - [Imports and Installs](#imports-and-installs)
  - [Paths and Hyperparameters](#paths-and-hyperparameters)
  - [Dataset and Collation](#dataset-and-collation)
  - [Training, Testing, and Saving](#training-testing-and-saving)
  - [Testing and Evaluation](#testing-and-evaluation)
  - [Final Model and Processor Saving](#final-model-and-processor-saving)
- [Results](#results)
- [Why These Libraries?](#why-these-libraries)

## Overview
This repository contains an action detection model using deep learning techniques with MediaPipe and TensorFlow. The model is trained to recognize different actions from video frames and achieves an accuracy score of 0.8.

## Technologies and Libraries Used
- Python
- TensorFlow/Keras
- MediaPipe
- OpenCV
- Scikit-learn
- Matplotlib
- NumPy

## How to Run
1. Install dependencies:
   ```sh
   pip install matplotlib scikit-learn mediapipe opencv-python tensorflow
   ```
2. Run the Jupyter Notebook or execute the script:
   ```sh
   python action_detection.py
   ```

## Detailed Code Explanation

### Imports and Installs
The code imports necessary libraries for deep learning, video processing, and data handling.

### Paths and Hyperparameters
The model uses predefined paths for datasets and initializes essential hyperparameters for training.

### Dataset and Collation
- Uses MediaPipe Holistic model for keypoint extraction.
- Processes video frames and extracts action-related keypoints.
- Labels the extracted data for supervised learning.

### Training, Testing, and Saving
- The dataset is split into training and testing sets.
- A Sequential deep learning model with LSTM and Dense layers is trained.
- The model is saved for future inference.

### Testing and Evaluation
- Predictions are evaluated using confusion matrices and accuracy scores.
- Accuracy achieved: **0.8 (80%)**.

### Final Model and Processor Saving
- The trained model is saved as a `.h5` file.
- The feature processor is stored for consistent preprocessing during inference.

## Results
- The action detection model successfully classifies different actions with 80% accuracy.
- Confusion matrices indicate good performance across multiple categories.

## Why These Libraries?
- **TensorFlow/Keras**: Provides deep learning capabilities.
- **MediaPipe**: Extracts human pose keypoints efficiently.
- **OpenCV**: Processes video frames.
- **Scikit-learn**: Evaluates model performance.
- **Matplotlib**: Visualizes model metrics.

