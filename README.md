# AI-Powered-Detection-of-Fetal-Distress
Datathon Team 83
Cardiotocography (CTG) Fetal Distress Prediction Model

Members' Names: Yan Xiaoye, Teh Zi En, Eleanor Lim Rui En




**Project Structure and Artifacts:**
The project is divided into distinct stages for data processing, training, inference, and reporting.


**Data_exploration_notebook.ipynb:**

Contains the initial data loading, cleanup steps (handling missing values/duplicates), exploratory analysis, and key visualizations. Review this first. Created By Raw data, Dependencies. Loaded By The user (initial review)


**train.py**

Master Training Script. Implements the full pipeline (Imputer, Scaler, SMOTE), compares 8 classifiers, and selects the best model based on Pathologic Recall. Creates the model and split data. Created By Raw data, Dependencies. Loaded By The user (initial run)


**test (1).py**

Final Inference Script. Loads the model artifact (.joblib file) and the unseen test data (data_split/), generates final metrics, and creates detailed Confusion Matrix and Feature Importance visualizations. Created By The user. Loaded By model_weights/best_ctg_model.joblib


**Academic Report.docx**

The final project report outlining the methodology, model selection rationale, ethical considerations, and key performance metrics. Created By The user. Loaded By Reviewer


**model_weights/best_ctg_model.joblib:**

The Trained Model Weight. The saved serialized ImbPipeline object containing the best-performing classifier and all preprocessing steps. Created By train.py. Loaded By test (1).py


**data_split/:**

Stores the final X_test.csv, y_test.csv, and X_train.csv files, saved during the training process for reproducibility. Created By train.py. Loaded By test (1).py






**Execution Flow :**
The project is to be run in two sequential steps to ensure the inference script has the necessary trained model and test data.

**Training & Model Saving:**
Run the master training script to select the best model and save the necessary files.

python train.py

Output: This script will print the model comparison table, train the final best model, save the split data to data_split/, and save the model to model_weights/best_ctg_model.joblib.

**Testing & Analysis:**
Run the inference script to load the saved model and evaluate its performance on the hold-out test set.

python "test (1).py"

Output: This will print the final inference report, including the critical safety metrics, and display the Confusion Matrix and Feature Importance plots.
