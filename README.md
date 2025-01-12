# Project 2 Machine Learning as part of the Higher Diploma in Computer Science and Data Analytics ATU 
# Seal Call Analysis and Classification

## Objective
The objective of this project is to analyze a recorded dataset of seal calls and investigate whether it is possible to discriminate between different types of calls (e.g., Rupe A, Rupe B, or no-call). The ultimate goal is to build a machine learning model capable of detecting and classifying seal calls based on their spectrogram representations.

## Project Overview
This project is divided into three main steps:

### 1. Data Preprocessing and Management

- **Input Data**: The provided uncompressed `.wav` files and their annotations.
- **Goal**: Create a dataset of spectrograms for seal calls and non-calls.
- **Process**:
  - Extract spectrograms for each annotated seal call.
  - Normalize the size of spectrograms based on the longest call and broadest frequency range.
  - Create additional spectrograms for non-call regions of the audio.
  - Save spectrograms as raw 2D arrays (not as images) in `.npz` format, along with metadata.
  - Ensure non-call spectrograms are extracted from the same frequency regions as the calls.
- **Tools Used**:
  - `matplotlib`
  - `scipy`
  - `numpy`
  - `pandas`

### 2. Model Training
- **Input Data**: `.npz` files containing spectrograms of calls and non-calls.
- **Goal**: Train a convolutional neural network (CNN) to classify seal calls.
- **Process**:
  - Perform data augmentation to improve model robustness.
  - Experiment with various hyperparameters and CNN architectures.
  - Evaluate the model's performance using a test set.
- **Tools Used**:
  - `tensorflow`/`keras`
  - `scikit-learn`
  - `matplotlib`
  - `numpy`

### 3. Refinement
- **Approach**:
  - Tune spectrogram parameters (e.g., `nfft`, `noverlap`) to improve resolution.
  - Split `.wav` files for easier processing if computational cost becomes prohibitive.
  - Validate the approach using similar datasets, such as bird call classification datasets from Kaggle.
  - Test the model by running it on an entire `.wav` file held back during training, ensuring no misclassification of overlapping calls.
- **Resources**:
  - [Kaggle Bird Call Dataset](https://www.kaggle.com/code/sophiagnetneva/cnn-for-sound-classificationbird-calls-90)

## File Structure
```
.
├── Part1_DataProcessing.ipynb  # Preprocesses audio files and generates spectrograms.
├── Part2_CNN_model.ipynb       # Builds, trains, and evaluates the CNN model.
├── data                        # Folder for raw and preprocessed data.
│   ├── raw_audio/              # Contains raw uncompressed `.wav` files.
│   ├── spectrograms/           # Stores generated spectrogram `.npz` files.
├── models                      # Folder to save trained models.
├── README.md                   # Project documentation.
```

## Part 1: Data Preprocessing
- Extract and preprocess spectrograms from raw `.wav` files.
- Normalize the spectrogram size.
- Save spectrograms as `.npz` files with metadata.

### Key Packages
```python
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
import glob
import os
from scipy.signal import spectrogram as compute_spectrogram
```

## Part 2: Model Training
- Load preprocessed spectrograms from `.npz` files.
- Perform data augmentation.
- Build and train a CNN model to classify seal calls.
- Experiment with different architectures and hyperparameters.

### CNN Architecture
- Convolutional layers for feature extraction.
- MaxPooling layers for dimensionality reduction.
- Dense layers for classification.

### Key Packages
```python
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from collections import Counter
import shutil
import random
```

## Getting Started
1. Clone this repository.
2. Install the required dependencies listed in `requirements.txt`.
3. Place the raw `.wav` files in the `data/raw_audio/` directory.
4. Run `Part1_DataProcessing.ipynb` to preprocess the data and generate spectrograms.
5. Run `Part2_CNN_model.ipynb` to train and evaluate the model.




