import os
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================
# HYPERPARAMETERS (USER CAN MODIFY)
# ============================================

# Model architecture
HIDDEN_LAYERS = [128,64]  # MLP hidden layers
ACTIVATION = "relu"
DROPOUT_RATE = 0.5

# Training parameters
TEST_SIZE = 0.2
VAL_SIZE = 0.25

# Hyperparameter search candidates
SEARCH_CONFIGS = [
    {"name": "tiny_32", "hidden_layers": [32], "dropout_rate": 0.1, "batch_size": 128, "epochs": 80, "learning_rate": 1e-3, "weight_decay": 1e-4, "early_stopping_patience": 10, "activation": "relu"},
    {"name": "tiny_64", "hidden_layers": [64], "dropout_rate": 0.2, "batch_size": 128, "epochs": 100, "learning_rate": 1e-3, "weight_decay": 1e-4, "early_stopping_patience": 12, "activation": "relu"},
    {"name": "tiny_64_32", "hidden_layers": [64, 32], "dropout_rate": 0.2, "batch_size": 128, "epochs": 100, "learning_rate": 2e-3, "weight_decay": 1e-4, "early_stopping_patience": 12, "activation": "relu"},
    {"name": "tiny_128_32", "hidden_layers": [128, 32], "dropout_rate": 0.25, "batch_size": 128, "epochs": 100, "learning_rate": 1e-3, "weight_decay": 1e-4, "early_stopping_patience": 12, "activation": "leaky_relu"},
    {"name": "tiny_deep", "hidden_layers": [128, 64, 32, 16], "dropout_rate": 0.25, "batch_size": 128, "epochs": 120, "learning_rate": 5e-4, "weight_decay": 1e-4, "early_stopping_patience": 15, "activation": "relu"},

    {"name": "baseline_relu", "hidden_layers": [128, 64], "dropout_rate": 0.3, "batch_size": 64, "epochs": 120, "learning_rate": 1e-3, "weight_decay": 1e-4, "early_stopping_patience": 15, "activation": "relu"},
    {"name": "baseline_leaky", "hidden_layers": [128, 64], "dropout_rate": 0.3, "batch_size": 64, "epochs": 120, "learning_rate": 1e-3, "weight_decay": 1e-4, "early_stopping_patience": 15, "activation": "leaky_relu"},
    {"name": "baseline_gelu", "hidden_layers": [128, 64], "dropout_rate": 0.3, "batch_size": 64, "epochs": 120, "learning_rate": 1e-3, "weight_decay": 1e-4, "early_stopping_patience": 15, "activation": "gelu"},
    {"name": "baseline_low_dropout", "hidden_layers": [128, 64], "dropout_rate": 0.1, "batch_size": 64, "epochs": 120, "learning_rate": 1e-3, "weight_decay": 1e-4, "early_stopping_patience": 15, "activation": "relu"},
    {"name": "baseline_high_dropout", "hidden_layers": [128, 64], "dropout_rate": 0.6, "batch_size": 64, "epochs": 150, "learning_rate": 5e-4, "weight_decay": 5e-4, "early_stopping_patience": 20, "activation": "relu"},

    {"name": "medium_256_128", "hidden_layers": [256, 128], "dropout_rate": 0.3, "batch_size": 64, "epochs": 150, "learning_rate": 5e-4, "weight_decay": 1e-4, "early_stopping_patience": 18, "activation": "relu"},
    {"name": "medium_256_128_64", "hidden_layers": [256, 128, 64], "dropout_rate": 0.35, "batch_size": 64, "epochs": 150, "learning_rate": 5e-4, "weight_decay": 1e-4, "early_stopping_patience": 18, "activation": "leaky_relu"},
    {"name": "medium_256_256_128", "hidden_layers": [256, 256, 128], "dropout_rate": 0.3, "batch_size": 64, "epochs": 160, "learning_rate": 3e-4, "weight_decay": 1e-4, "early_stopping_patience": 20, "activation": "gelu"},
    {"name": "medium_384_192_96", "hidden_layers": [384, 192, 96], "dropout_rate": 0.3, "batch_size": 64, "epochs": 160, "learning_rate": 5e-4, "weight_decay": 2e-4, "early_stopping_patience": 20, "activation": "leaky_relu"},
    {"name": "medium_512_128", "hidden_layers": [512, 128], "dropout_rate": 0.35, "batch_size": 64, "epochs": 180, "learning_rate": 3e-4, "weight_decay": 1e-4, "early_stopping_patience": 20, "activation": "relu"},

    {"name": "bottleneck_512_256_128", "hidden_layers": [512, 256, 128], "dropout_rate": 0.25, "batch_size": 64, "epochs": 150, "learning_rate": 5e-4, "weight_decay": 1e-4, "early_stopping_patience": 18, "activation": "leaky_relu"},
    {"name": "bottleneck_512_128_32", "hidden_layers": [512, 128, 32], "dropout_rate": 0.3, "batch_size": 64, "epochs": 160, "learning_rate": 3e-4, "weight_decay": 2e-4, "early_stopping_patience": 20, "activation": "gelu"},
    {"name": "bottleneck_1024_256_64", "hidden_layers": [1024, 256, 64], "dropout_rate": 0.35, "batch_size": 32, "epochs": 180, "learning_rate": 2e-4, "weight_decay": 2e-4, "early_stopping_patience": 20, "activation": "leaky_relu"},
    {"name": "hourglass_256_64_256", "hidden_layers": [256, 64, 256], "dropout_rate": 0.3, "batch_size": 64, "epochs": 160, "learning_rate": 3e-4, "weight_decay": 1e-4, "early_stopping_patience": 18, "activation": "gelu"},
    {"name": "hourglass_512_128_512_128", "hidden_layers": [512, 128, 512, 128], "dropout_rate": 0.35, "batch_size": 32, "epochs": 180, "learning_rate": 2e-4, "weight_decay": 2e-4, "early_stopping_patience": 22, "activation": "leaky_relu"},

    {"name": "wide_512_512", "hidden_layers": [512, 512], "dropout_rate": 0.3, "batch_size": 64, "epochs": 180, "learning_rate": 3e-4, "weight_decay": 1e-4, "early_stopping_patience": 20, "activation": "relu"},
    {"name": "wide_512_512_256", "hidden_layers": [512, 512, 256], "dropout_rate": 0.35, "batch_size": 32, "epochs": 180, "learning_rate": 3e-4, "weight_decay": 1e-4, "early_stopping_patience": 20, "activation": "leaky_relu"},
    {"name": "wide_768_384_192", "hidden_layers": [768, 384, 192], "dropout_rate": 0.35, "batch_size": 32, "epochs": 200, "learning_rate": 2e-4, "weight_decay": 2e-4, "early_stopping_patience": 22, "activation": "gelu"},
    {"name": "wide_1024_512_256", "hidden_layers": [1024, 512, 256], "dropout_rate": 0.4, "batch_size": 32, "epochs": 200, "learning_rate": 2e-4, "weight_decay": 2e-4, "early_stopping_patience": 22, "activation": "leaky_relu"},
    {"name": "wide_1024_1024_512", "hidden_layers": [1024, 1024, 512], "dropout_rate": 0.4, "batch_size": 32, "epochs": 220, "learning_rate": 1e-4, "weight_decay": 3e-4, "early_stopping_patience": 25, "activation": "gelu"},

    {"name": "deep_256", "hidden_layers": [256, 256, 256, 256], "dropout_rate": 0.35, "batch_size": 32, "epochs": 200, "learning_rate": 2e-4, "weight_decay": 2e-4, "early_stopping_patience": 22, "activation": "relu"},
    {"name": "deep_funnel", "hidden_layers": [512, 384, 256, 128, 64], "dropout_rate": 0.35, "batch_size": 32, "epochs": 220, "learning_rate": 2e-4, "weight_decay": 2e-4, "early_stopping_patience": 25, "activation": "leaky_relu"},
    {"name": "deep_gelu", "hidden_layers": [512, 512, 256, 256, 128], "dropout_rate": 0.4, "batch_size": 32, "epochs": 220, "learning_rate": 1e-4, "weight_decay": 3e-4, "early_stopping_patience": 25, "activation": "gelu"},
    {"name": "deep_constant_512", "hidden_layers": [512, 512, 512, 512], "dropout_rate": 0.4, "batch_size": 32, "epochs": 240, "learning_rate": 1e-4, "weight_decay": 3e-4, "early_stopping_patience": 25, "activation": "leaky_relu"},
    {"name": "deep_zigzag", "hidden_layers": [512, 256, 512, 256, 128], "dropout_rate": 0.4, "batch_size": 32, "epochs": 220, "learning_rate": 1e-4, "weight_decay": 2e-4, "early_stopping_patience": 25, "activation": "gelu"},

    {"name": "regularized_light", "hidden_layers": [512, 256, 128], "dropout_rate": 0.45, "batch_size": 64, "epochs": 200, "learning_rate": 3e-4, "weight_decay": 5e-4, "early_stopping_patience": 20, "activation": "relu"},
    {"name": "regularized_medium", "hidden_layers": [1024, 512, 256], "dropout_rate": 0.5, "batch_size": 32, "epochs": 220, "learning_rate": 2e-4, "weight_decay": 1e-3, "early_stopping_patience": 25, "activation": "leaky_relu"},
    {"name": "regularized_hard", "hidden_layers": [1024, 512, 256, 128], "dropout_rate": 0.6, "batch_size": 32, "epochs": 250, "learning_rate": 1e-4, "weight_decay": 5e-3, "early_stopping_patience": 30, "activation": "gelu"},
    {"name": "almost_no_regularization", "hidden_layers": [1024, 512, 256], "dropout_rate": 0.05, "batch_size": 64, "epochs": 180, "learning_rate": 3e-4, "weight_decay": 1e-6, "early_stopping_patience": 20, "activation": "leaky_relu"},
    {"name": "no_dropout_test", "hidden_layers": [512, 256, 128], "dropout_rate": 0.0, "batch_size": 64, "epochs": 180, "learning_rate": 5e-4, "weight_decay": 1e-4, "early_stopping_patience": 18, "activation": "relu"},

    {"name": "small_batch_8", "hidden_layers": [512, 256, 128], "dropout_rate": 0.35, "batch_size": 8, "epochs": 180, "learning_rate": 1e-4, "weight_decay": 2e-4, "early_stopping_patience": 20, "activation": "leaky_relu"},
    {"name": "small_batch_16", "hidden_layers": [1024, 512, 256], "dropout_rate": 0.4, "batch_size": 16, "epochs": 200, "learning_rate": 1e-4, "weight_decay": 2e-4, "early_stopping_patience": 22, "activation": "gelu"},
    {"name": "large_batch_128", "hidden_layers": [512, 256, 128], "dropout_rate": 0.3, "batch_size": 128, "epochs": 200, "learning_rate": 1e-3, "weight_decay": 1e-4, "early_stopping_patience": 20, "activation": "relu"},
    {"name": "large_batch_256", "hidden_layers": [1024, 512, 256], "dropout_rate": 0.35, "batch_size": 256, "epochs": 220, "learning_rate": 2e-3, "weight_decay": 2e-4, "early_stopping_patience": 25, "activation": "leaky_relu"},
    {"name": "very_large_batch", "hidden_layers": [2048, 1024, 512], "dropout_rate": 0.4, "batch_size": 512, "epochs": 250, "learning_rate": 2e-3, "weight_decay": 3e-4, "early_stopping_patience": 25, "activation": "gelu"},

    {"name": "high_lr_test", "hidden_layers": [256, 128, 64], "dropout_rate": 0.35, "batch_size": 64, "epochs": 120, "learning_rate": 5e-3, "weight_decay": 1e-4, "early_stopping_patience": 10, "activation": "relu"},
    {"name": "very_high_lr_test", "hidden_layers": [128, 64], "dropout_rate": 0.4, "batch_size": 128, "epochs": 100, "learning_rate": 1e-2, "weight_decay": 1e-4, "early_stopping_patience": 8, "activation": "relu"},
    {"name": "low_lr_test", "hidden_layers": [1024, 512, 256], "dropout_rate": 0.3, "batch_size": 32, "epochs": 300, "learning_rate": 5e-5, "weight_decay": 1e-4, "early_stopping_patience": 30, "activation": "leaky_relu"},
    {"name": "very_low_lr_test", "hidden_layers": [2048, 1024, 512, 256], "dropout_rate": 0.3, "batch_size": 32, "epochs": 400, "learning_rate": 1e-5, "weight_decay": 1e-4, "early_stopping_patience": 35, "activation": "gelu"},

    {"name": "deli_1453", "hidden_layers": [2048, 1024, 512, 256, 128], "dropout_rate": 0.25, "batch_size": 64, "epochs": 200, "learning_rate": 5e-4, "weight_decay": 1e-4, "early_stopping_patience": 20, "activation": "leaky_relu"},
    {"name": "deli_1453_v2", "hidden_layers": [4096, 2048, 1024, 512, 256, 128], "dropout_rate": 0.25, "batch_size": 64, "epochs": 400, "learning_rate": 1e-5, "weight_decay": 1e-4, "early_stopping_patience": 15, "activation": "leaky_relu"},
    {"name": "deli_1453_v3", "hidden_layers": [4096, 4096, 2048, 1024, 512, 256], "dropout_rate": 0.4, "batch_size": 16, "epochs": 350, "learning_rate": 5e-5, "weight_decay": 3e-4, "early_stopping_patience": 30, "activation": "gelu"},
    {"name": "deli_1453_v4", "hidden_layers": [8192, 4096, 2048, 1024, 512, 256, 128], "dropout_rate": 0.5, "batch_size": 8, "epochs": 400, "learning_rate": 2e-5, "weight_decay": 5e-4, "early_stopping_patience": 35, "activation": "leaky_relu"},
    {"name": "deli_1453_v5", "hidden_layers": [8192, 8192, 4096, 2048, 1024, 512], "dropout_rate": 0.55, "batch_size": 4, "epochs": 500, "learning_rate": 1e-5, "weight_decay": 1e-3, "early_stopping_patience": 40, "activation": "gelu"},
    {"name": "deli_1453_hourglass", "hidden_layers": [4096, 1024, 256, 1024, 4096, 512], "dropout_rate": 0.5, "batch_size": 8, "epochs": 400, "learning_rate": 2e-5, "weight_decay": 5e-4, "early_stopping_patience": 35, "activation": "leaky_relu"},
    {"name": "deli_1453_zigzag", "hidden_layers": [2048, 512, 2048, 512, 1024, 256, 128], "dropout_rate": 0.45, "batch_size": 16, "epochs": 350, "learning_rate": 5e-5, "weight_decay": 3e-4, "early_stopping_patience": 30, "activation": "gelu"}
]

# Global seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================================
# OUTPUT DIRECTORY
# ============================================

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("MLP Training - Raw Images (48x48x3)")
print("=" * 60)
print(f"\nHYPERPARAMETERS:")
print(f"  Hidden layers: {HIDDEN_LAYERS}")
print(f"  Activation: {ACTIVATION}")
print(f"  Dropout: {DROPOUT_RATE}")
print("Hyperparameter search candidates:")
for cfg in SEARCH_CONFIGS:
    print(f"  - {cfg['name']}: hidden={cfg['hidden_layers']}, dropout={cfg['dropout_rate']}, batch={cfg['batch_size']}, epochs={cfg['epochs']}, lr={cfg['learning_rate']}, wd={cfg['weight_decay']}")

# ============================================
# 1. LOAD DATA
# ============================================

print("\n" + "=" * 60)
print("1. LOAD DATA")
print("=" * 60)

# Load images.npy with memory mapping and labels.txt
images = np.load("images.npy", mmap_mode="r")  # Shape: (N, 224, 224, 3)
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

print(f"Images memory-mapped: {images.shape}")
print(f"Labels loaded: {len(labels)} samples")
print(f"First 5 labels: {labels[:5]}")
print(f"Unique labels: {len(np.unique(labels))}")

# ============================================
# 2. DATA PREPROCESSING & INDEX SPLITTING
# ============================================

print("\n" + "=" * 60)
print("2. DATA PREPROCESSING & INDEX SPLITTING")
print("=" * 60)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(np.unique(encoded_labels))

print(f"Labels encoded: {num_classes} classes")

# Train-Test split on indices
indices = np.arange(len(images))
train_val_idx, test_idx, y_train_val, y_test = train_test_split(
    indices,
    encoded_labels,
    test_size=TEST_SIZE,
    stratify=encoded_labels,
    random_state=42,
    shuffle=True,
)

# Train-Val split on indices
train_idx, val_idx, y_train, y_val = train_test_split(
    train_val_idx,
    y_train_val,
    test_size=VAL_SIZE,
    random_state=42,
    shuffle=True,
)

print(f"Train: {len(train_idx)} ({100*len(train_idx)/len(images):.1f}%)")
print(f"Val:   {len(val_idx)} ({100*len(val_idx)/len(images):.1f}%)")
print(f"Test:  {len(test_idx)} ({100*len(test_idx)/len(images):.1f}%)")

# Memory-efficient chunked computation of mean & std on Train Set
def compute_mean_std_chunked(mmap_images, indices, chunk_size=500):
    total_sum = 0.0
    total_sq_sum = 0.0
    num_elements = 0
    img_size = np.prod(mmap_images.shape[1:])
    
    for i in range(0, len(indices), chunk_size):
        chunk_indices = indices[i : i + chunk_size]
        chunk = mmap_images[chunk_indices].astype(np.float64)
        total_sum += chunk.sum()
        total_sq_sum += (chunk ** 2).sum()
        num_elements += len(chunk_indices) * img_size
        
    mean = total_sum / num_elements
    var = (total_sq_sum / num_elements) - (mean ** 2)
    std = np.sqrt(max(var, 0.0))
    return mean, std

def compute_min_max_chunked(mmap_images, indices, mean, std, chunk_size=500):
    min_val = float('inf')
    max_val = float('-inf')
    for i in range(0, len(indices), chunk_size):
        chunk_indices = indices[i : i + chunk_size]
        chunk = mmap_images[chunk_indices].astype(np.float32)
        chunk_norm = (chunk - mean) / (std + 1e-8)
        min_val = min(min_val, chunk_norm.min())
        max_val = max(max_val, chunk_norm.max())
    return min_val, max_val

print("Calculating normalization statistics from Train set...")
mean, std = compute_mean_std_chunked(images, train_idx, chunk_size=500)
min_val, max_val = compute_min_max_chunked(images, train_idx, mean, std)
print(f"Data normalized: mean={mean:.4f}, std={std:.4f}")
print(f"After norm (Train) - min={min_val:.4f}, max={max_val:.4f}")

# Define the custom Dataset
class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, mmap_images, indices, labels, mean, std):
        self.mmap_images = mmap_images
        self.indices = indices
        self.labels = labels
        self.mean = mean
        self.std = std
        self.input_dim = np.prod(mmap_images.shape[1:])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img = np.array(self.mmap_images[real_idx], dtype=np.float32).flatten()
        img_norm = (img - self.mean) / (self.std + 1e-8)
        label = self.labels[idx]
        return torch.tensor(img_norm, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

train_dataset = FaceDataset(images, train_idx, y_train, mean, std)
val_dataset = FaceDataset(images, val_idx, y_val, mean, std)
test_dataset = FaceDataset(images, test_idx, y_test, mean, std)

print(f"DataLoaders and Datasets created")

# ============================================
# 4. BUILD MLP MODEL
# ============================================

print("\n" + "=" * 60)
print("4. BUILD MLP MODEL")
print("=" * 60)

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_layers, num_classes, dropout_rate=0.3):
        super(MLPModel, self).__init__()
        layers = []
        
        # Input -> First hidden
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        layers.append(nn.BatchNorm1d(hidden_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden -> Hidden
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.BatchNorm1d(hidden_layers[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Last hidden -> Output
        layers.append(nn.Linear(hidden_layers[-1], num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

input_dim = train_dataset.input_dim
model = MLPModel(input_dim, HIDDEN_LAYERS, num_classes, DROPOUT_RATE)
model = model.to(DEVICE)

print("\nModel architecture:")
print(model)

# ============================================
# 5. TRAIN WITH HYPERPARAMETER SEARCH
# ============================================

print("\n" + "=" * 60)
print("5. HYPERPARAMETER SEARCH")
print("=" * 60)

loss_fn = nn.CrossEntropyLoss()

# Build data loaders once for the fixed splits
# (train_dataset, val_dataset, test_dataset are already initialized)

best_result = None

for config in SEARCH_CONFIGS:
    print(f"\nRunning trial: {config['name']}")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    model = MLPModel(input_dim, config["hidden_layers"], num_classes, config["dropout_rate"])
    model = model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                outputs = model(X_batch)
                loss = loss_fn(outputs, y_batch)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"  Epoch [{epoch + 1}/{config['epochs']}] - "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
        )

        acc_gap = train_acc - val_acc
        if acc_gap > 0.15:
            print(f"    possible overfitting (gap={acc_gap:.3f})")

        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-6 or (abs(val_loss - best_val_loss) < 1e-6 and val_acc > best_val_acc):
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    model.eval()

    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            test_total += y_batch.size(0)
            test_correct += (predicted == y_batch).sum().item()

    test_acc = test_correct / test_total

    train_correct_final = 0
    train_total_final = 0
    with torch.no_grad():
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            train_total_final += y_batch.size(0)
            train_correct_final += (predicted == y_batch).sum().item()

    train_acc_final = train_correct_final / train_total_final

    print(f"  Result -> val_acc={val_acc:.4f}, test_acc={test_acc:.4f}, train_acc={train_acc_final:.4f}")

    trial_result = {
        "name": config["name"],
        "config": config,
        "state_dict": best_state,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "train_acc": train_acc_final,
    }

    if best_result is None or trial_result["val_acc"] > best_result["val_acc"] + 1e-6:
        best_result = trial_result

print(f"\nBest trial: {best_result['name']}")
print(f"Best validation accuracy: {best_result['val_acc']:.4f}")

# ============================================
# 6. LOAD BEST MODEL AND TEST
# ============================================

print("\n" + "=" * 60)
print("6. TEST RESULTS")
print("=" * 60)

best_model = MLPModel(input_dim, best_result["config"]["hidden_layers"], num_classes, best_result["config"]["dropout_rate"])
best_model = best_model.to(DEVICE)
best_model.load_state_dict(best_result["state_dict"])
best_model.eval()

# Re-evaluate best model on train/val/test
val_loader = DataLoader(val_dataset, batch_size=best_result["config"]["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=best_result["config"]["batch_size"], shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=best_result["config"]["batch_size"], shuffle=False)

val_correct = 0
val_total = 0
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        outputs = best_model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        val_total += y_batch.size(0)
        val_correct += (predicted == y_batch).sum().item()
val_acc = val_correct / val_total

test_correct = 0
test_total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        outputs = best_model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        test_total += y_batch.size(0)
        test_correct += (predicted == y_batch).sum().item()
test_acc = test_correct / test_total

train_correct_final = 0
train_total_final = 0
with torch.no_grad():
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        outputs = best_model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        train_total_final += y_batch.size(0)
        train_correct_final += (predicted == y_batch).sum().item()
train_acc_final = train_correct_final / train_total_final

print(f"\nBest Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Train Accuracy: {train_acc_final:.4f}")

# ============================================
# 7. PLOTS
# ============================================

print("\n" + "=" * 60)
print("7. GENERATING PLOTS")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(best_result["train_accs"], label="Train", linewidth=2)
axes[0].plot(best_result["val_accs"], label="Validation", linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Model Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(best_result["train_losses"], label="Train", linewidth=2)
axes[1].plot(best_result["val_losses"], label="Validation", linewidth=2)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].set_title("Model Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "training_history.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Plot saved: {OUTPUT_DIR / 'training_history.png'}")

# ============================================
# 8. SAVE MODEL
# ============================================

print("\n" + "=" * 60)
print("8. SAVE MODEL")
print("=" * 60)

torch.save(best_model.state_dict(), OUTPUT_DIR / "mlp_model.pth")
print(f"Model saved: {OUTPUT_DIR / 'mlp_model.pth'}")

torch.save(best_model.state_dict(), OUTPUT_DIR / "best_model.pth")
print(f"Best model saved: {OUTPUT_DIR / 'best_model.pth'}")

import pickle
with open(OUTPUT_DIR / "label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
print(f"Label encoder saved: {OUTPUT_DIR / 'label_encoder.pkl'}")

# ============================================
# 9. SAVE SUMMARY
# ============================================

with open(OUTPUT_DIR / "training_summary.txt", "w") as f:
    f.write("=" * 60 + "\n")
    f.write("MLP Training Summary\n")
    f.write("=" * 60 + "\n\n")

    f.write("BEST CONFIGURATION:\n")
    f.write(f"  Name: {best_result['name']}\n")
    f.write(f"  Hidden layers: {best_result['config']['hidden_layers']}\n")
    f.write(f"  Dropout: {best_result['config']['dropout_rate']}\n")
    f.write(f"  Batch size: {best_result['config']['batch_size']}\n")
    f.write(f"  Epochs: {best_result['config']['epochs']}\n")
    f.write(f"  Learning rate: {best_result['config']['learning_rate']}\n")
    f.write(f"  Weight decay: {best_result['config']['weight_decay']}\n\n")

    f.write("DATA:\n")
    f.write(f"  Train: {len(train_idx)}\n")
    f.write(f"  Val: {len(val_idx)}\n")
    f.write(f"  Test: {len(test_idx)}\n")
    f.write(f"  Classes: {num_classes}\n\n")

    f.write("RESULTS:\n")
    f.write(f"  Validation Accuracy: {val_acc:.4f}\n")
    f.write(f"  Train Accuracy: {train_acc_final:.4f}\n")
    f.write(f"  Test Accuracy: {test_acc:.4f}\n")

print(f"Summary saved: {OUTPUT_DIR / 'training_summary.txt'}")

print("\n" + "=" * 60)
print("ALL DONE!")
print("=" * 60)

            