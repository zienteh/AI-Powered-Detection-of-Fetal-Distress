import subprocess
import sys
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, recall_score, f1_score, balanced_accuracy_score
from sklearn.tree import plot_tree # Import for visualizing the tree structure

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
    'pandas': 'pandas', 'scikit-learn': 'sklearn', 'matplotlib': 'matplotlib',
    'xgboost': 'xgboost', 'imbalanced-learn': 'imblearn' # imblearn needed for ImbPipeline loading
}
for package, import_name in required_packages.items():
    install_and_import(package, import_name)


# Set consistent styling for plots
plt.style.use('ggplot')

# --- 1. CONFIGURATION AND LOADING ARTIFACTS ---

DATA_DIR = 'data_split'
MODEL_DIR = 'model_weights'
MODEL_PATH = f'{MODEL_DIR}/final_best_model.joblib'

try:
    print("--- 1. Loading Test Data and Final Trained Model ---")
    # Load test data saved by data_exploration/data_prep.py
    X_test = pd.read_csv(f'{DATA_DIR}/X_test.csv')
    y_test = pd.read_csv(f'{DATA_DIR}/y_test.csv')['Fetal_Status'] # Load as Series

    # Load the best trained pipeline (saved by train_model.py)
    final_pipeline = joblib.load(MODEL_PATH)
    best_model_name = final_pipeline.named_steps['classifier'].__class__.__name__

    # Map the technical class name back to the project name if possible
    if 'GradientBoostingClassifier' in best_model_name:
        best_model_display_name = 'GradientBoostingClassifier'
    elif 'XGBClassifier' in best_model_name:
        best_model_display_name = 'XGBoostClassifier'
    else:
        best_model_display_name = best_model_name
        
    print(f"Loaded test data ({len(X_test)} samples).")
    print(f"Loaded final trained model: {best_model_display_name}.")

except FileNotFoundError as e:
    print(f"\nFATAL ERROR: Required file not found: {e.filename}.")
    print("Please ensure you have run data_prep.py AND train_model.py successfully to create all required files.")
    sys.exit(1)


# --- 2. FINAL PREDICTION AND METRIC CALCULATION ---

# Predict on the hold-out test set
y_pred = final_pipeline.predict(X_test)

# Define target names (Labels are 0, 1, 2)
target_names = ['Normal (0)', 'Suspect (1)', 'Pathologic (2)']

# 2.1 Calculate Final Metrics (using the same scorers as the master script)
# Note: Pathologic label is 2
final_pathologic_recall = recall_score(y_test, y_pred, labels=[2], average='macro', zero_division=0)
final_f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
final_balanced_acc = balanced_accuracy_score(y_test, y_pred)


# --- 3. FINAL MODEL INFERENCE REPORT (CONSOLE OUTPUT) ---

metrics_data = [
    ("Pathologic Recall", final_pathologic_recall, "Minimizes missed fetal distress cases (Safety Focus)"),
    ("F1 Macro Score", final_f1_macro, "Overall balance between Precision and Recall"),
    ("Balanced Accuracy", final_balanced_acc, "Overall performance accounting for class imbalance")
]

print("\n" + "=" * 55)
print("{:^55}".format("FINAL MODEL INFERENCE REPORT"))
print("=" * 55)
print(f"Model Used: {best_model_display_name} (Best-Performing Classifier)\n")

# Print the fixed-width console output table
print("| {:<20} | {:<10} | {:<50} |".format("Metric", "Score", "Interpretation"))
print("|{:-<22}|{:-<12}|{:-<52}|".format("", "", "")) # Separator line with exact column widths

for metric_name, score, interpretation in metrics_data:
    # Use f-string for formatting with fixed widths and 4 decimal places
    print(f"| {metric_name:<20} | {score:^10.4f} | {interpretation:<50} |")

print("\n(Metrics are calculated on the 30% hold-out test set.)")


# --- 4. VISUALIZATION AND INTERPRETATION ---

# 4.1 Classification Report (Detailed Breakdown)
print(f"\n--- 4.1 Detailed Classification Report for Final Model ({best_model_display_name}) ---")
print(classification_report(y_test, y_pred, target_names=target_names))


# 4.2 Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix for Final Model: {best_model_display_name} (Goal: High Recall for Pathologic)')
plt.show(block=False) # Show plot non-blocking


# 4.3 Feature Importance & Tree Visualization (Only for Tree-based models)
classifier_step = final_pipeline.named_steps['classifier']
is_tree_model = hasattr(classifier_step, 'feature_importances_')

if is_tree_model:
    print(f"\n--- 4.3 Feature Importance & Structure for Clinical Understanding ({best_model_display_name}) ---")

    # Access the trained classifier step inside the pipeline
    model = classifier_step
    importances = model.feature_importances_
    # Use original feature names from the test set
    feature_names = X_test.columns
    sorted_indices = np.argsort(importances)

    # --- A. Feature Importance Plot ---
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importance for {best_model_display_name}")
    plt.barh(range(X_test.shape[1]), importances[sorted_indices], align='center')
    plt.yticks(range(X_test.shape[1]), feature_names[sorted_indices])
    plt.xlabel("Relative Importance Score")
    plt.show(block=False)

    # --- B. Decision Tree Structure Visualization (for DecisionTree only) ---
    if 'DecisionTreeClassifier' in best_model_name:
        print("\n--- Decision Tree Structure Visualization (Max Depth 3) ---")
        tree_model = classifier_step
        plt.figure(figsize=(18, 10))
        # Plot the first few decision nodes (max_depth=3) for clarity
        plot_tree(tree_model,
                  feature_names=feature_names,
                  class_names=target_names,
                  filled=True,
                  rounded=True,
                  fontsize=8,
                  max_depth=3)
        plt.title(f'Decision Tree Visualization (First 3 Levels) - {best_model_display_name}')
        plt.show(block=True)
    else:
        # Show Feature Importance plot last if the winner is not a single Decision Tree
        plt.show(block=True)

else:
    print(f"\nFeature Importance Plot skipped: Not applicable for selected model ({best_model_display_name}).")

print("\n--- Master Datathon Analysis Complete ---")