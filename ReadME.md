Cardiotocography (CTG) Fetal Distress Prediction Model

Team Information:
Team: TM-83
Members' Names: Yan Xiaoye, Teh Zi En, Eleanor Lim Rui En

Project Structure and Artifacts:
The project is divided into distinct stages for data processing, training, inference, and reporting.


Data_exploration_notebook.ipynb:

Contains the initial data loading, cleanup steps (handling missing values/duplicates), exploratory analysis, and key visualizations. Review this first. Created By Raw data, Dependencies. Loaded By The user (initial review)



train.py

Master Training Script. Implements the full pipeline (Imputer, Scaler, SMOTE), compares 8 classifiers, and selects the best model based on Pathologic Recall. Creates the model and split data. Created By Raw data, Dependencies. Loaded By The user (initial run)



test (1).py

Final Inference Script. Loads the model artifact (.joblib file) and the unseen test data (data_split/), generates final metrics, and creates detailed Confusion Matrix and Feature Importance visualizations. Created By The user. Loaded By model_weights/best_ctg_model.joblib



Academic Report.docx

Document (Report)

The final project report outlining the methodology, model selection rationale, ethical considerations, and key performance metrics. Created By The user. Loaded By Reviewer



model_weights/best_ctg_model.joblib

Binary File (Model Artifact)

The Trained Model Weight. The saved serialized ImbPipeline object containing the best-performing classifier and all preprocessing steps. Created By train.py. Loaded By test (1).py




data_split/:

Directory (Data Artifacts)

Stores the final X_test.csv, y_test.csv, and X_train.csv files, saved during the training process for reproducibility. Created By train.py. Loaded By test (1).py