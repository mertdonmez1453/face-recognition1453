# Face Recognition Project

This repository contains a face recognition experiment using EfficientNet feature extraction and K-Nearest Neighbors classification.

## Project structure

- `train.py` - Trains and evaluates a KNN classifier using PCA and cached features.
- `preprocess.py` - Loads images, extracts PyTorch EfficientNetB0 features, and caches `features.npy` and `labels.txt`.
- `datasets/Faces-Datasets-scaled/` - Input face image folders organized by person name.
- `features.npy` - Cached extracted feature vectors.
- `images.npy` - Cached normalized images (optional).
- `labels.txt` - Cached label names corresponding to features.

## Requirements

Recommended Python packages:

- `numpy`
- `opencv-python`
- `scikit-learn`
- `matplotlib`
- `torch`
- `torchvision`

Install dependencies with pip:

```bash
pip install numpy opencv-python scikit-learn matplotlib torch torchvision
```

## Usage

1. Prepare the dataset under `datasets/Faces-Datasets/` with one subfolder per person.
   - `train.py` will automatically resize images into `datasets/Faces-Datasets-scaled/` if needed.
2. Run `train.py`:

```bash
python train.py
```

The script will:

- Load cached `features.npy` and `labels.txt` if available.
- Normalize features and split the data into train / validation / test sets.
- Search over different `k` values and PCA variance ratios.
- Print validation scores and the final test performance.
- Display the confusion matrix.

## Notes

- `preprocess.py` uses EfficientNetB0 to extract features from images.
- If cached features are missing, the script will build them automatically.
- Hyperparameters can be changed in `train.py` by editing `k_values` and `pca_variances`.
