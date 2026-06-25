import os
import cv2
import numpy as np
from preprocess import features, labels

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import normalize

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# normalize the features
features = normalize(features)

print(f"Loaded features: {features.shape}")
print(f"Loaded labels: {len(labels)}")



##split dataset into train and test
X_train_val, X_test, y_train_val, y_test = train_test_split(features,labels, test_size=0.2, stratify=labels, random_state=42, shuffle=True)


# split train+val into train (75% of 80% = 60%) and validation (25% of 80% = 20%)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.25,          # 25% of train+val becomes validation
    random_state=42,
    shuffle=True
)

print(f"\nData split:")
print(f"  Train:      {X_train.shape[0]} samples ({100*X_train.shape[0]/features.shape[0]:.1f}%)")
print(f"  Validation: {X_val.shape[0]} samples ({100*X_val.shape[0]/features.shape[0]:.1f}%)")
print(f"  Test:       {X_test.shape[0]} samples ({100*X_test.shape[0]/features.shape[0]:.1f}%)")

# Save original splits for repeated experiments
X_train_orig = X_train.copy()
X_val_orig = X_val.copy()
X_test_orig = X_test.copy()

k_values = [1, 3, 5]
pca_variances = [0.55, 0.90, 0.95, 0.97, 0.99]

best_val_f1 = -1.0
best_k = None
best_pca_ratio = None

for k in k_values:
    for pca_ratio in pca_variances:
        pca = PCA(n_components=pca_ratio)
        X_train = pca.fit_transform(X_train_orig)
        X_val = pca.transform(X_val_orig)

        knn_clf = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='minkowski')
        knn_clf.fit(X_train, y_train)

        y_val_pred = knn_clf.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
        val_recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)

        print(f"\n=== k={k}, PCA variance={pca_ratio:.2f} ===")
        print(f"    Validation accuracy:  {val_accuracy:.4f}")
        print(f"    Validation precision: {val_precision:.4f}")
        print(f"    Validation recall:    {val_recall:.4f}")
        print(f"    Validation F1:        {val_f1:.4f}")
        print(f"    k value:   {k}")
        print(f"    PCA ratio: {X_val.shape}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_k = k
            best_pca_ratio = pca_ratio

print(f"\n=== Best hyperparameters from validation ===")
print(f"    Best k:         {best_k}")
print(f"    Best PCA ratio: {best_pca_ratio}")
print(f"    Best val F1:    {best_val_f1:.4f}")

# Final test run using the best hyperparameters once
pca = PCA(n_components=best_pca_ratio)
X_train = pca.fit_transform(X_train_orig)
X_test = pca.transform(X_test_orig)

knn_clf = KNeighborsClassifier(n_neighbors=best_k, weights='distance', metric='minkowski')
knn_clf.fit(X_train, y_train)
final_y_pred = knn_clf.predict(X_test)

final_accuracy = accuracy_score(y_test, final_y_pred)
final_precision = precision_score(y_test, final_y_pred, average='weighted', zero_division=0)
final_recall = recall_score(y_test, final_y_pred, average='weighted', zero_division=0)
final_f1 = f1_score(y_test, final_y_pred, average='weighted', zero_division=0)

print(f"\n=== Final test result (best k/pca) ===")
print(f"    k:               {best_k}")
print(f"    PCA ratio:       {best_pca_ratio}")
print(f"    Test accuracy:   {final_accuracy:.4f}")
print(f"    Test precision:  {final_precision:.4f}")
print(f"    Test recall:     {final_recall:.4f}")
print(f"    Test F1:         {final_f1:.4f}")


cm = confusion_matrix(y_test, final_y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(xticks_rotation=90)

plt.show()

##feature cikarmak icin resnet yerine effifecent net kullandım modellerden cıkan outputu degistirimiyorum cunku zaten pcade dusuruyoruz
##pca orani degisebeilir
##k sayisi degisebilir 