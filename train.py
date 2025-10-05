import subprocess
import sys
import time
import os
import joblib
import pandas as pd
import numpy as np

# --- 0. INSTALLATION CHECK AND SETUP ---
def install_and_import(package, import_name=None):
    """Checks if a package is installed, and if not, attempts to install it via pip."""
    if import_name is None:
        import_name = package
    try:
        __import__(import_name)
    except ImportError:
        print(f"'{import_name}' not found. Attempting to install '{package}'...")
        try:
            # Install without printing stdout to keep the console clean
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Successfully installed '{package}'.")
            __import__(import_name)
        except subprocess.CalledProcessError as e:
            print(f"Error installing '{package}'. Please run 'pip install {package}' manually.")
            print(f"Details: {e}")
            sys.exit(1)

# List of critical packages (using their pip install name)
required_packages = {
    'pandas': 'pandas', 'scikit-learn': 'sklearn', 'imbalanced-learn': 'imblearn',
    'xgboost': 'xgboost', 'joblib': 'joblib'
}
for package, import_name in required_packages.items():
    install_and_import(package, import_name)

# --- Standard Imports ---
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, make_scorer, f1_score, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Classifier Imports
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb


# --- 1. CONFIGURATION AND LOADING DATA ---

DATA_DIR = 'data_split'
MODEL_DIR = 'model_weights'
os.makedirs(MODEL_DIR, exist_ok=True) # Ensure model directory exists

try:
    print("--- 1. Loading Pre-split Data for Training ---")
    # Load data saved by the data preparation script
    # The 'data_split' directory must exist and contain the CSV files.
    X_train = pd.read_csv(f'{DATA_DIR}/X_train.csv')
    y_train = pd.read_csv(f'{DATA_DIR}/y_train.csv')['Fetal_Status'] # Load as Series
    print(f"Loaded training data: {len(X_train)} samples.")
except FileNotFoundError as e:
    print(f"\nFATAL ERROR: Required data file not found: {e.filename}.")
    print("Please ensure you run the data preparation script first to create the 'data_split/' directory and files.")
    sys.exit(1)


# --- 2. MODEL COMPARISON PIPELINES & SCORING (Evaluation Focus) ---

# CRITICAL SCORER: Define a scorer to prioritize RECALL for the Pathologic class (label 2).
def pathologic_recall_scorer(y_true, y_pred):
    """Calculates Recall specifically for the Pathologic class (label 2)."""
    # Calculate Recall only for the Pathologic class (now label 2)
    return recall_score(y_true, y_pred, labels=[2], average='macro', zero_division=0)

pathologic_recall = make_scorer(pathologic_recall_scorer)

# Define Preprocessing steps common to all models: Imputation -> Scaling
preprocessing_steps = [
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
]

# Define all 8 Model Classifiers
models_to_compare = {
    'LogReg': LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42, class_weight='balanced'),
    'DecisionTree': DecisionTreeClassifier(max_depth=7, random_state=42, class_weight='balanced'),
    'RandomForest': RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, class_weight='balanced'),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    'SVC': SVC(kernel='rbf', gamma='auto', probability=True, random_state=42, class_weight='balanced'),
    'KNeighbors': KNeighborsClassifier(n_neighbors=5),
    'NeuralNet': MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, objective='multi:softmax', num_class=3)
}

pipelines = {}
for name, classifier in models_to_compare.items():
    # Use SMOTE within the pipeline for all models
    pipelines[name] = ImbPipeline(steps=preprocessing_steps + [
        ('smote', SMOTE(random_state=42, k_neighbors=3)),
        ('classifier', classifier)
    ])


# --- 3. CROSS-VALIDATION AND SELECTION (Evaluation Focus) ---

print("\n--- 3. Cross-Validating All Models (Scoring: Recall for Pathologic Class) ---")
results_df = pd.DataFrame(columns=['Model', 'Pathologic Recall (CV)', 'F1 Macro (CV)', 'Balanced Accuracy (CV)', 'Time (s)', 'Pipeline'])

for name, pipeline in pipelines.items():
    start_time = time.time()

    # Perform 5-fold cross-validation on the three primary metrics
    cv_pathologic_recall = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=pathologic_recall, n_jobs=-1).mean()
    cv_f1_macro = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1).mean()
    cv_balanced_acc = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='balanced_accuracy', n_jobs=-1).mean()

    end_time = time.time()

    # Store results
    results_df.loc[len(results_df)] = [
        name,
        cv_pathologic_recall,
        cv_f1_macro,
        cv_balanced_acc,
        round(end_time - start_time, 2),
        pipeline
    ]
    print(f"Model: {name} | Path. Recall: {cv_pathologic_recall:.4f} | F1 Macro: {cv_f1_macro:.4f} | Time: {results_df.iloc[-1]['Time (s)']}s")

# Find the Best Model based on Pathologic Recall
best_model_data = results_df.sort_values(by='Pathologic Recall (CV)', ascending=False).iloc[0]
best_model_name = best_model_data['Model']
final_pipeline = best_model_data['Pipeline']

print(f"\n--- BEST MODEL SELECTED (Minimize False Negatives): {best_model_name} (Path. Recall: {best_model_data['Pathologic Recall (CV)']:.4f}) ---")

# Print the comparison table
print("\n--- Comprehensive Model Comparison ---")
print(results_df[['Model', 'Pathologic Recall (CV)', 'F1 Macro (CV)', 'Balanced Accuracy (CV)', 'Time (s)']]\
      .sort_values(by='Pathologic Recall (CV)', ascending=False).to_markdown(index=False))


# --- 4. FINAL TRAINING AND SAVING ---

print("\n--- 4. Training Final Model on Full Training Set ---")
# Train the best pipeline on the full training set (including SMOTE inside the pipeline)
final_pipeline.fit(X_train, y_train)

# Save the final trained pipeline as joblib
MODEL_PATH = f'{MODEL_DIR}/final_best_model.joblib'
joblib.dump(final_pipeline, MODEL_PATH)

print(f"--- Successfully saved final model to: {MODEL_PATH} ---")
print("--- Training Complete. Run test_inference.py next. ---")