import os
import cv2
import numpy as np

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False
else:
    try:
        from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
    except Exception:
        TF_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


def load_images_to_tensor(input_dir, image_size=(224, 224), normalize=True):
    images = []
    labels = []
    paths = []

    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    for person_name in sorted(os.listdir(input_dir)):
        person_path = os.path.join(input_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        for img_name in sorted(os.listdir(person_path)):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if (img.shape[0], img.shape[1]) != image_size:
                img = cv2.resize(img, image_size)

            img = img.astype(np.float32)
            if normalize:
                img /= 255.0

            images.append(img)
            labels.append(person_name)
            paths.append(img_path)

    if len(images) == 0:
        return np.empty((0, image_size[0], image_size[1], 3), dtype=np.float32), [], []

    return np.stack(images, axis=0), labels, paths


def images_to_efficientnet_features(images, batch_size=32, model=None):
    if TORCH_AVAILABLE:
        try:
            import torch.nn as nn
            from torchvision.models import efficientnet_b0

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if model is None:
                try:
                    from torchvision.models import EfficientNet_B0_Weights
                    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
                    base = efficientnet_b0(weights=weights)
                except Exception:
                    base = efficientnet_b0(pretrained=True)
                feature_extractor = nn.Sequential(*list(base.children())[:-1])
                feature_extractor.to(device)
                feature_extractor.eval()
            else:
                feature_extractor = model.to(device)

            imgs_t = torch.from_numpy(images).permute(0, 3, 1, 2).to(dtype=torch.float32, device=device)
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            imgs_t = (imgs_t - mean) / std

            feats_list = []
            with torch.no_grad():
                for i in range(0, imgs_t.shape[0], batch_size):
                    batch = imgs_t[i:i + batch_size]
                    out = feature_extractor(batch)
                    out = out.view(out.size(0), -1)
                    feats_list.append(out.cpu().numpy())

            if not feats_list:
                return np.empty((0, 1280), dtype=np.float32)
            return np.vstack(feats_list)
        except Exception as e:
            raise RuntimeError(f"PyTorch EfficientNet extraction failed: {e}")

    if not TF_AVAILABLE:
        raise RuntimeError("Neither PyTorch nor TensorFlow is available")

    try:
        if model is None:
            model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    except Exception as e:
        raise RuntimeError(f"Failed to load EfficientNet model: {e}")

    imgs = (images * 255.0).astype(np.float32)
    imgs = preprocess_input(imgs)
    return model.predict(imgs, batch_size=batch_size, verbose=1)


def _load_labels(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def _save_labels(path, labels):
    with open(path, 'w', encoding='utf-8') as f:
        for label in labels:
            f.write(label + "\n")


def build_images(source_dir="datasets/Faces-Datasets-scaled"):
    imgs, labels, paths = load_images_to_tensor(source_dir, normalize=True)
    np.save("images.npy", imgs)
    _save_labels("labels.txt", labels)
    return imgs, labels


def build_features(source_dir="datasets/Faces-Datasets-scaled"):
    imgs, labels = build_images(source_dir)
    feats = images_to_efficientnet_features(imgs)
    np.save("features.npy", feats)
    return feats, labels


def load_cached_images(images_path="images.npy", labels_path="labels.txt"):
    if os.path.exists(images_path) and os.path.exists(labels_path):
        return np.load(images_path), _load_labels(labels_path)
    return None, None


def load_cached_features(features_path="features.npy", labels_path="labels.txt"):
    if os.path.exists(features_path) and os.path.exists(labels_path):
        return np.load(features_path), _load_labels(labels_path)
    return None, None


# Keep old-style feature data caching for compatibility
features, labels = load_cached_features()
if features is None:
    print("Caching features for the first time. This may take a while...")
    features, labels = build_features()


if __name__ == "__main__":
    print(f"Loaded features shape: {features.shape}")
    print(f"Loaded labels: {len(labels)}")



