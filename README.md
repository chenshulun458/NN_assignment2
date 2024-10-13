# Abalone Age Prediction Project

This project predicts the age of abalone using various models (Linear Regression, Logistic Regression, Neural Networks) and combines them using a fusion approach. The structure is organized into data, experiments, notebooks, and source code.

## Project Structure

- **data/**
  - `X_train.csv` : Training features dataset.
  - `X_test.csv` : Test features dataset.
  - `y_train.csv` : Training labels dataset.
  - `y_test.csv` : Test labels dataset.
  - `README.md` : Dataset description or instructions.

- **experiments/**
  - `accuracy_comparison.png` : Accuracy comparison between models.
  - `advanced_model_comparison.png` : Comparison of advanced model performance.
  - `auc_comparison.png` : AUC score comparison of models.
  - `comparison_combined.png` : Combined performance comparison of models.
  - `r2_comparison.png` : R-squared comparison between models.
  - `rmse_comparison.png` : Root Mean Squared Error comparison between models.
  - `imag1.png` : Additional image related to model results.
  - `model_output.txt` : Output file containing model performance summaries.

- **notebooks/**
  - `exploratory_analysis.ipynb` : Notebook for data exploration and visualization.
  - `model_development.ipynb` : Notebook for developing the machine learning models.
  - `model_hyperparameter_tuning.ipynb` : Notebook for hyperparameter tuning of the models.

- **src/**
  - `data_processing.py` : Script for data preprocessing and cleaning.
  - `experiments_visualization_1.py` : Script for generating visualizations from model experiments.
  - `experiments_visualization_2.py` : Additional visualization script for comparing experiments.
  - `feature_engineer.py` : Script for creating and engineering new features for the model.
  - `model.py` : Script containing model definitions and training processes.

- **.gitignore** : Specifies files and directories to be ignored by git.
- **requirements.txt** : List of dependencies and Python packages needed for the project.

