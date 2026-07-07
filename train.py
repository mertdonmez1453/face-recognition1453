import os
import cv2
import numpy as np
from pathlib import Path
from preprocess import features, labels

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import normalize, LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_confusion_matrix_png(y_true, y_pred, class_names, filename):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="viridis")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix: {output_path}")


# normalize the features
features = normalize(features)

print(f"Loaded features: {features.shape}")
print(f"Loaded labels: {len(labels)}")



##split dataset into train and test
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

X_train_val, X_test, y_train_val, y_test = train_test_split(
    features,
    encoded_labels,
    test_size=0.2,
    stratify=encoded_labels,
    random_state=42,
    shuffle=True,
)

# split train+val into train (75% of 80% = 60%) and validation (25% of 80% = 20%)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    shuffle=True,
)

print(f"\nData split:")
print(f"  Train:      {X_train.shape[0]} samples ({100*X_train.shape[0]/features.shape[0]:.1f}%)")
print(f"  Validation: {X_val.shape[0]} samples ({100*X_val.shape[0]/features.shape[0]:.1f}%)")
print(f"  Test:       {X_test.shape[0]} samples ({100*X_test.shape[0]/features.shape[0]:.1f}%)")

# Save original splits for repeated experiments
X_train_orig = X_train.copy()
X_val_orig = X_val.copy()
X_test_orig = X_test.copy()


KNN_train = True

if KNN_train == True:

    k_values = [1]
    pca_variances = [0.97]

    best_val_f1 = -1.0
    best_k = None
    best_pca_ratio = None

    for k in k_values:
        for pca_ratio in pca_variances:
            pca = PCA(n_components=pca_ratio)
            X_train = pca.fit_transform(X_train_orig)  # Fit PCA on training data
            X_val = pca.transform(X_val_orig)          # Transform validation data

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
    X_train = pca.fit_transform(X_train_orig)  # Fit PCA on training data
    X_test = pca.transform(X_test_orig)        # Transform test data

    knn_clf = KNeighborsClassifier(n_neighbors=best_k, weights='distance', metric='minkowski')
    knn_clf.fit(X_train, y_train)
    final_y_pred = knn_clf.predict(X_test)

    final_accuracy = accuracy_score(y_test, final_y_pred)
    final_precision = precision_score(y_test, final_y_pred, average='weighted', zero_division=0)
    final_recall = recall_score(y_test, final_y_pred, average='weighted', zero_division=0)
    final_f1 = f1_score(y_test, final_y_pred, average='weighted', zero_division=0)

    print(f"\n=== Final test result KNN model (best k/pca) ===")
    print(f"    k:               {best_k}")
    print(f"    PCA ratio:       {best_pca_ratio}")
    print(f"    Test accuracy:   {final_accuracy:.4f}")
    print(f"    Test precision:  {final_precision:.4f}")
    print(f"    Test recall:     {final_recall:.4f}")
    print(f"    Test F1:         {final_f1:.4f}")

    save_confusion_matrix_png(
        y_test,
        final_y_pred,
        label_encoder.classes_,
        "knn_confusion_matrix.png",
    )



### decison tree model training

treeTrain = True

if treeTrain == True:
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    # Try different hyperparameters
    criterions = ['entropy']
    max_depths = [500]
    min_samples_leafs = [1]
    pca_components = [0.90]
    best_val_f1 = -1.0
    best_params = None
    best_tree_model = None

    print("\n=== Decision Tree Hyperparameter Tuning ===")
    total_combinations = len(criterions) * len(max_depths) * len(min_samples_leafs) * len(pca_components)
    print("Tuning criterion, max_depth, min_samples_leaf, and PCA components")
    print(f"Total combinations: {total_combinations}")
    print(f"PCA values: {pca_components}")
    print(f"Max depths: {max_depths}")
    print(f"Min samples leaf: {min_samples_leafs}")
    print()

    counter = 0
    for pca_ratio in pca_components:
        pca = PCA(n_components=pca_ratio)
        X_train = pca.fit_transform(X_train_orig)  # Fit PCA on training data
        X_val = pca.transform(X_val_orig)          # Transform validation data
        X_test = pca.transform(X_test_orig)        # Transform test data


        for criterion in criterions:
            for depth in max_depths:
                for min_leaf in min_samples_leafs:
                    counter += 1
                    tree_clf = DecisionTreeClassifier(
                        criterion=criterion,
                        max_depth=depth,
                        min_samples_leaf=min_leaf,
                        random_state=42,
                    )
                    tree_clf.fit(X_train, y_train)

                    y_val_pred = tree_clf.predict(X_val)
                    val_accuracy = accuracy_score(y_val, y_val_pred)
                    val_precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
                    val_recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
                    val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)

                    result_text = f"[{counter:5d}/{total_combinations}] PCA={pca_ratio:.2f} | criterion={criterion:10s} | depth={depth:3d} | min_leaf={min_leaf} | F1: {val_f1:.4f} | Acc: {val_accuracy:.4f}"
                    print(result_text)

                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_params = {
                            'pca_ratio': pca_ratio,
                            'criterion': criterion,
                            'max_depth': depth,
                            'min_samples_leaf': min_leaf
                        }
                        best_tree_model = tree_clf
                        best_X_test = X_test

    best_text = f"""
PCA ratio:          {best_params['pca_ratio']:.2f}
Criterion:          {best_params['criterion']}
Max depth:          {best_params['max_depth']}
Min samples leaf:   {best_params['min_samples_leaf']}
Best val F1:        {best_val_f1:.4f}
"""
    print(best_text)

    # Final test evaluation with best model
    y_test_pred = best_tree_model.predict(best_X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

    final_text = f"""
{"-" * 120}
Final Test Results
{"-" * 120}
PCA ratio:          {best_params['pca_ratio']:.2f}
Criterion:          {best_params['criterion']}
Max depth:          {best_params['max_depth']}
Min samples leaf:   {best_params['min_samples_leaf']}
Test accuracy:      {test_accuracy:.4f}
Test precision:     {test_precision:.4f}
Test recall:        {test_recall:.4f}
Test F1:            {test_f1:.4f}
{"-" * 120}
"""
    print(final_text)

    save_confusion_matrix_png(
        y_test,
        y_test_pred,
        label_encoder.classes_,
        "decision_tree_confusion_matrix.png",
    )



randomForestTrain = True

if randomForestTrain == True:

    from sklearn.ensemble import RandomForestClassifier

    # Try different hyperparameters
    max_features = ['sqrt']
    n_estimators = [700]
    max_depths = [100]
    pca_components = [0.90]
    best_val_f1 = -1.0
    best_params = None
    best_rf_model = None

    print("\n=== Random Forest Hyperparameter Tuning ===")
    total_combinations = len(max_features) * len(n_estimators) * len(max_depths) * len(pca_components)
    print("Tuning max_features, n_estimators, max_depth, and PCA components")
    print(f"Total combinations: {total_combinations}")
    print(f"PCA values: {pca_components}")
    print(f"Max depths: {max_depths}")
    print(f"N estimators: {n_estimators}")
    print(f"Max features: {max_features}")
    print()

    counter = 0
    for pca_ratio in pca_components:
        pca = PCA(n_components=pca_ratio)
        X_train = pca.fit_transform(X_train_orig)  # Fit PCA on training data
        X_val = pca.transform(X_val_orig)          # Transform validation data
        X_test = pca.transform(X_test_orig)        # Transform test data

        # Normalize after PCA
        X_train_norm = normalize(X_train)
        X_val_norm = normalize(X_val)
        X_test_norm = normalize(X_test)

        for max_feat in max_features:
            for n_est in n_estimators:
                for depth in max_depths:
                    counter += 1
                    rf_clf = RandomForestClassifier(
                        max_features=max_feat,
                        n_estimators=n_est,
                        max_depth=depth,
                        random_state=42,
                        n_jobs=-1
                    )
                    rf_clf.fit(X_train_norm, y_train)
                    
                    y_val_pred = rf_clf.predict(X_val_norm)
                    val_accuracy = accuracy_score(y_val, y_val_pred)
                    val_precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
                    val_recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
                    val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
                    
                    result_text = f"[{counter:5d}/{total_combinations}] PCA={pca_ratio:.2f} | max_feat={str(max_feat):6s} | n_est={n_est:3d} | depth={depth:3d} | F1: {val_f1:.4f} | Acc: {val_accuracy:.4f}"
                    print(result_text)
                    
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_params = {
                            'pca_ratio': pca_ratio,
                            'max_features': max_feat,
                            'n_estimators': n_est,
                            'max_depth': depth
                        }
                        best_rf_model = rf_clf
                        best_X_test = X_test_norm

    best_text = f"""
PCA ratio:       {best_params['pca_ratio']:.2f}
Max features:    {best_params['max_features']}
N estimators:    {best_params['n_estimators']}
Max depth:       {best_params['max_depth']}
Best val F1:     {best_val_f1:.4f}
"""
    print(best_text)

    # Final test evaluation with best model
    y_test_pred = best_rf_model.predict(best_X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

    final_text = f"""
{"-" * 120}
Final Test Results
{"-" * 120}
PCA ratio:       {best_params['pca_ratio']:.2f}
Max features:    {best_params['max_features']}
N estimators:    {best_params['n_estimators']}
Max depth:       {best_params['max_depth']}
Test accuracy:   {test_accuracy:.4f}
Test precision:  {test_precision:.4f}
Test recall:     {test_recall:.4f}
Test F1:         {test_f1:.4f}
{"-" * 120}
"""
    print(final_text)

    save_confusion_matrix_png(
        y_test,
        y_test_pred,
        label_encoder.classes_,
        "random_forest_confusion_matrix.png",
    )


gradientBoostingTrain = True

if gradientBoostingTrain == True:
    try:
        import xgboost as xgb
        from xgboost import XGBClassifier
        xgb_available = True
    except Exception as e:
        xgb_available = False
        print(f"XGBoost unavailable, falling back to sklearn: {e}")

    if xgb_available:
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except Exception:
            gpu_available = False

        tree_method = 'hist'
        device = 'cuda' if gpu_available else 'cpu'

        learning_rates = [0.1]
        max_depths = [3]
        pca_components = [0.90]
        best_val_f1 = -1.0
        best_params = None
        best_gb_model = None

        print("\n=== Gradient Boosting Hyperparameter Tuning (XGBoost) ===")
        total_combinations = len(learning_rates) * len(max_depths) * len(pca_components)
        print("Tuning learning_rate, max_depth, and PCA components")
        print(f"Total combinations: {total_combinations}")
        print(f"PCA values: {pca_components}")
        print(f"Max depths: {max_depths}")
        print(f"Learning rates: {learning_rates}")
        print(f"Training backend: {device} ({tree_method})")
        print()

        counter = 0
        for pca_ratio in pca_components:
            pca = PCA(n_components=pca_ratio)
            X_train = pca.fit_transform(X_train_orig)
            X_val = pca.transform(X_val_orig)
            X_test = pca.transform(X_test_orig)

            X_train_norm = normalize(X_train)
            X_val_norm = normalize(X_val)
            X_test_norm = normalize(X_test)

            for lr in learning_rates:
                for depth in max_depths:
                    counter += 1
                    gb_clf = XGBClassifier(
                        learning_rate=lr,
                        max_depth=depth,
                        n_estimators=200,
                        objective='multi:softprob',
                        eval_metric='mlogloss',
                        random_state=42,
                        tree_method=tree_method,
                        device=device,
                        n_jobs=4,
                    )
                    gb_clf.fit(X_train_norm, y_train)

                    y_val_pred = gb_clf.predict(X_val_norm)
                    val_accuracy = accuracy_score(y_val, y_val_pred)
                    val_precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
                    val_recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
                    val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)

                    result_text = f"[{counter:6d}/{total_combinations}] PCA={pca_ratio:.2f} | lr={lr:.2f} | depth={depth:2d} | F1: {val_f1:.4f} | Acc: {val_accuracy:.4f}"
                    print(result_text)

                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_params = {
                            'pca_ratio': pca_ratio,
                            'learning_rate': lr,
                            'max_depth': depth
                        }
                        best_gb_model = gb_clf
                        best_X_test = X_test_norm

        best_text = f"""
PCA ratio:       {best_params['pca_ratio']:.2f}
Learning rate:   {best_params['learning_rate']}
Max depth:       {best_params['max_depth']}
Best val F1:     {best_val_f1:.4f}
"""
        print(best_text)

        y_test_pred = best_gb_model.predict(best_X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

        final_text = f"""
{"-" * 140}
Final Test Results
{"-" * 140}
PCA ratio:           {best_params['pca_ratio']:.2f}
Learning rate:       {best_params['learning_rate']:.2f}
Max depth:           {best_params['max_depth']}
Test accuracy:       {test_accuracy:.4f}
Test precision:      {test_precision:.4f}
Test recall:         {test_recall:.4f}
Test F1:             {test_f1:.4f}
{"-" * 140}
"""
        print(final_text)

        save_confusion_matrix_png(
            y_test,
            y_test_pred,
            label_encoder.classes_,
            "gradient_boosting_confusion_matrix.png",
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier

        learning_rates = [0.01, 0.05, 0.2]
        max_depths = [2, 6]
        pca_components = [0.50, 0.90, 0.99]
        best_val_f1 = -1.0
        best_params = None
        best_gb_model = None

        print("\n=== Gradient Boosting Hyperparameter Tuning (sklearn fallback) ===")
        total_combinations = len(learning_rates) * len(max_depths) * len(pca_components)
        print("Tuning learning_rate, max_depth, and PCA components")
        print(f"Total combinations: {total_combinations}")
        print(f"PCA values: {pca_components}")
        print(f"Max depths: {max_depths}")
        print(f"Learning rates: {learning_rates}")
        print()

        counter = 0
        for pca_ratio in pca_components:
            pca = PCA(n_components=pca_ratio)
            X_train = pca.fit_transform(X_train_orig)
            X_val = pca.transform(X_val_orig)
            X_test = pca.transform(X_test_orig)

            X_train_norm = normalize(X_train)
            X_val_norm = normalize(X_val)
            X_test_norm = normalize(X_test)

            for lr in learning_rates:
                for depth in max_depths:
                    counter += 1
                    gb_clf = GradientBoostingClassifier(
                        learning_rate=lr,
                        n_estimators=100,
                        max_depth=depth,
                        random_state=42
                    )
                    gb_clf.fit(X_train_norm, y_train)

                    y_val_pred = gb_clf.predict(X_val_norm)
                    val_accuracy = accuracy_score(y_val, y_val_pred)
                    val_precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
                    val_recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
                    val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)

                    result_text = f"[{counter:6d}/{total_combinations}] PCA={pca_ratio:.2f} | lr={lr:.2f} | depth={depth:2d} | F1: {val_f1:.4f} | Acc: {val_accuracy:.4f}"
                    print(result_text)

                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_params = {
                            'pca_ratio': pca_ratio,
                            'learning_rate': lr,
                            'max_depth': depth
                        }
                        best_gb_model = gb_clf
                        best_X_test = X_test_norm

        best_text = f"""
PCA ratio:       {best_params['pca_ratio']:.2f}
Learning rate:   {best_params['learning_rate']}
Max depth:       {best_params['max_depth']}
Best val F1:     {best_val_f1:.4f}
"""
        print(best_text)

        y_test_pred = best_gb_model.predict(best_X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

        final_text = f"""
{"-" * 140}
Final Test Results
{"-" * 140}
PCA ratio:           {best_params['pca_ratio']:.2f}
Learning rate:       {best_params['learning_rate']:.2f}
Max depth:           {best_params['max_depth']}
Test accuracy:       {test_accuracy:.4f}
Test precision:      {test_precision:.4f}
Test recall:         {test_recall:.4f}
Test F1:             {test_f1:.4f}
{"-" * 140}
"""
        print(final_text)

