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


KNN_train = False

if KNN_train == True:

    k_values = [1, 3, 5]
    pca_variances = [0.55, 0.90, 0.95, 0.97, 0.99]

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



### decison tree model training

treeTrain = False

if treeTrain == True:
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    # Try different hyperparameters
    criterions = ['gini', 'entropy']
    splitters = ['best', 'random']
    max_depths = [20, 40, 60]
    min_samples_leafs = [4, 5]
    pca_components = [0.65, 0.85, 0.90, 0.99]
    best_val_f1 = -1.0
    best_params = None
    best_tree_model = None

    counter = 0
    for pca_ratio in pca_components:
        pca = PCA(n_components=pca_ratio)
        X_train = pca.fit_transform(X_train_orig)  # Fit PCA on training data
        X_val = pca.transform(X_val_orig)          # Transform validation data
        X_test = pca.transform(X_test_orig)        # Transform test data

        # NO NORMALIZATION - Decision Tree doesn't need it

        for criterion in criterions:
            for splitter in splitters:
                for depth in max_depths:
                    for min_leaf in min_samples_leafs:
                        counter += 1
                        tree_clf = DecisionTreeClassifier(
                            criterion=criterion,
                            splitter=splitter,
                            max_depth=depth,
                            min_samples_leaf=min_leaf,
                            random_state=42,
                        )
                        tree_clf.fit(X_train_orig, y_train)
                        
                        y_val_pred = tree_clf.predict(X_val_orig)
                        val_accuracy = accuracy_score(y_val, y_val_pred)
                        val_precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
                        val_recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
                        val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
                        
                        result_text = f"[{counter:5d}/{total_combinations}] PCA={pca_ratio:.2f} | criterion={criterion:10s} | splitter={splitter:6s} | depth={depth:3d} | min_leaf={min_leaf} | F1: {val_f1:.4f} | Acc: {val_accuracy:.4f}"
                        print(result_text)
                        
                        if val_f1 > best_val_f1:
                            best_val_f1 = val_f1
                            best_params = {
                                'pca_ratio': pca_ratio,
                                'criterion': criterion,
                                'splitter': splitter,
                                'max_depth': depth,
                                'min_samples_leaf': min_leaf
                            }
                            best_tree_model = tree_clf
                            best_X_test = X_test

    best_text = f"""
PCA ratio:          {best_params['pca_ratio']:.2f}
Criterion:          {best_params['criterion']}
Splitter:           {best_params['splitter']}
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
Splitter:           {best_params['splitter']}
Max depth:          {best_params['max_depth']}
Min samples leaf:   {best_params['min_samples_leaf']}
Test accuracy:      {test_accuracy:.4f}
Test precision:     {test_precision:.4f}
Test recall:        {test_recall:.4f}
Test F1:            {test_f1:.4f}
{"-" * 120}
"""
    print(final_text)



randomForestTrain = True

if randomForestTrain == True:

    from sklearn.ensemble import RandomForestClassifier

    # Try different hyperparameters
    max_features = ['sqrt', 'log2', None]
    n_estimators = [50,  150,  250, 300]
    max_depths = [10, 20, 30, 100]
    pca_components = [0.50, 0.90, 0.99]
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


gradientBoostingTrain = False

if gradientBoostingTrain == True:

    from sklearn.ensemble import GradientBoostingClassifier

    # Try different hyperparameters
    losses = ['log_loss', 'exponential']
    learning_rates = [0.01, 0.15, 0.2]
    n_estimators = [50, 150, 300]
    max_depths = [3, 7, 10]
    min_samples_splits = [2, 5, 10]
    pca_components = [0.50, 0.90, 0.99]
    best_val_f1 = -1.0
    best_params = None
    best_gb_model = None

    print("\n=== Gradient Boosting Hyperparameter Tuning ===")
    total_combinations = len(losses) * len(learning_rates) * len(n_estimators) * len(max_depths) * len(min_samples_splits) * len(pca_components)
    print("Tuning loss, learning_rate, n_estimators, max_depth, min_samples_split, and PCA components")
    print(f"Total combinations: {total_combinations}")
    print(f"PCA values: {pca_components}")
    print(f"Max depths: {max_depths}")
    print(f"N estimators: {n_estimators}")
    print(f"Learning rates: {learning_rates}")
    print(f"Losses: {losses}")
    print(f"Min samples split: {min_samples_splits}")
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

        for loss in losses:
            for lr in learning_rates:
                for n_est in n_estimators:
                    for depth in max_depths:
                        for min_split in min_samples_splits:
                            counter += 1
                            gb_clf = GradientBoostingClassifier(
                                loss=loss,
                                learning_rate=lr,
                                n_estimators=n_est,
                                max_depth=depth,
                                min_samples_split=min_split,
                                random_state=42
                            )
                            gb_clf.fit(X_train_norm, y_train)
                            
                            y_val_pred = gb_clf.predict(X_val_norm)
                            val_accuracy = accuracy_score(y_val, y_val_pred)
                            val_precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
                            val_recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
                            val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
                            
                            result_text = f"[{counter:6d}/{total_combinations}] PCA={pca_ratio:.2f} | loss={loss:11s} | lr={lr:.2f} | n_est={n_est:3d} | depth={depth:2d} | min_split={min_split:2d} | F1: {val_f1:.4f} | Acc: {val_accuracy:.4f}"
                            print(result_text)
                            
                            if val_f1 > best_val_f1:
                                best_val_f1 = val_f1
                                best_params = {
                                    'pca_ratio': pca_ratio,
                                    'loss': loss,
                                    'learning_rate': lr,
                                    'n_estimators': n_est,
                                    'max_depth': depth,
                                    'min_samples_split': min_split
                                }
                                best_gb_model = gb_clf
                                best_X_test = X_test_norm

    best_text = f"""
PCA ratio:       {best_params['pca_ratio']:.2f}
Loss:            {best_params['loss']}
Learning rate:   {best_params['learning_rate']}
N estimators:    {best_params['n_estimators']}
Max depth:       {best_params['max_depth']}
Min samples split: {best_params['min_samples_split']}
Best val F1:     {best_val_f1:.4f}
"""
    print(best_text)

    # Final test evaluation with best model
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
Loss:                {best_params['loss']}
Learning rate:       {best_params['learning_rate']:.2f}
N estimators:        {best_params['n_estimators']}
Max depth:           {best_params['max_depth']}
Min samples split:   {best_params['min_samples_split']}
Test accuracy:       {test_accuracy:.4f}
Test precision:      {test_precision:.4f}
Test recall:         {test_recall:.4f}
Test F1:             {test_f1:.4f}
{"-" * 140}
"""
    print(final_text)

