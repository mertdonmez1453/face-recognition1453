import os
import cv2
import numpy as np

from resize import resize_dataset

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


def _has_image_subfolders(directory):
    if not os.path.isdir(directory):
        return False

    for person_name in os.listdir(directory):
        person_path = os.path.join(directory, person_name)
        if not os.path.isdir(person_path):
            continue
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            if os.path.isfile(img_path):
                return True
    return False


def ensure_scaled_dataset(source_dir="datasets/Faces-Datasets", scaled_dir="datasets/Faces-Datasets-scaled"):
    if _has_image_subfolders(scaled_dir):
        return scaled_dir

    if not _has_image_subfolders(source_dir):
        raise ValueError(
            f"Source dataset directory does not contain valid images: {source_dir}. "
            "Please add unscaled images under datasets/Faces-Datasets or provide a scaled dataset."
        )

    os.makedirs(scaled_dir, exist_ok=True)
    print(f"Scaling dataset from {source_dir} to {scaled_dir}...")
    resize_dataset(source_dir, scaled_dir)
    return scaled_dir


def images_to_efficientnet_features(images, batch_size=8, model=None):
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

            # Define normalization constants
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

            feats_list = []
            with torch.no_grad():
                # Use mixed precision for memory efficiency on GPU
                use_amp = device.type == 'cuda'
                autocast = torch.cuda.amp.autocast if use_amp else lambda: torch.no_op()
                
                # Process in batches to avoid GPU memory overflow
                for i in range(0, images.shape[0], batch_size):
                    # Load batch to GPU
                    batch_np = images[i:i + batch_size]
                    batch_t = torch.from_numpy(batch_np).permute(0, 3, 1, 2).to(dtype=torch.float32, device=device)
                    
                    # Normalize batch
                    batch_t = (batch_t - mean) / std
                    
                    # Extract features with mixed precision on GPU
                    with autocast():
                        out = feature_extractor(batch_t)
                    out = out.view(out.size(0), -1).float()
                    feats_list.append(out.cpu().numpy())
                    
                    # Clear GPU cache between batches
                    if use_amp:
                        torch.cuda.empty_cache()
                    
                    # Progress indicator
                    processed = min(i + batch_size, images.shape[0])
                    print(f"  Processed {processed}/{images.shape[0]} images...", end='\r')

            print(f"  Processed {images.shape[0]}/{images.shape[0]} images - Done!        ")
            if not feats_list:
                return np.empty((0, 1280), dtype=np.float32)
            return np.vstack(feats_list)
        except Exception as e:
            raise RuntimeError(f"PyTorch EfficientNet extraction failed: {e}")

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for EfficientNet feature extraction")

    raise RuntimeError("PyTorch feature extraction fallback should not be reached")


def _load_labels(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def _save_labels(path, labels):
    with open(path, 'w', encoding='utf-8') as f:
        for label in labels:
            f.write(label + "\n")


def build_images(source_dir="datasets/Faces-Datasets-scaled"):
    if source_dir == "datasets/Faces-Datasets-scaled":
        source_dir = ensure_scaled_dataset()

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



