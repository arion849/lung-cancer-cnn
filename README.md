# Lung Cancer Detection with a CNN

A convolutional neural network, built with TensorFlow/Keras, that classifies lung histopathology images into three classes:

| Class | Meaning |
| --- | --- |
| `lung_aca` | Lung adenocarcinoma |
| `lung_scc` | Lung squamous cell carcinoma |
| `lung_n` | Benign lung tissue |

## Data

The model trains on the lung images from the [Lung and Colon Cancer Histopathological Images (LC25000)](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) dataset on Kaggle. The images aren't tracked in this repo. Download the dataset and set `path` in `CNN_model.py` to the `lung_image_sets` folder.

## Model

- Images resized to 256×256 RGB, 80/20 train/validation split
- Three Conv2D + MaxPooling blocks (32 → 64 → 128 filters)
- Dense 256 → 128 head with batch normalisation and dropout, softmax over 3 classes
- Adam optimiser, categorical cross-entropy
- Callbacks: early stopping on validation accuracy, learning-rate reduction on plateau, and a hard stop once validation accuracy passes 90%

The script plots sample images per class and the training curves, then prints a per-class classification report (precision, recall, F1).

## Run

```bash
pip install tensorflow opencv-python scikit-learn pandas matplotlib pillow
python CNN_model.py
```
