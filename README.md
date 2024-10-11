# NN_Assignment2

This repository contains the code and resources for the **Neural Network Assignment 2**. The project aims to perform data preprocessing, feature engineering, and model development using various neural network techniques. The goal is to predict the target variable using the provided dataset and evaluate model performance.

## Project Structure

The project is organized into the following directories and files:
NN_Assignment2/ ├── .venv/ # Python virtual environment ├── data/ # Directory for datasets │ ├── X_train.csv # Training data for features │ ├── X_test.csv # Testing data for features │ ├── y_train.csv # Training data for target variable │ └── y_test.csv # Testing data for target variable ├── experiments/ # Directory for storing experimental results ├── notebooks/ # Jupyter notebooks for exploratory analysis and visualization │ └── exploratory_analysis.ipynb ├── reports/ # Directory for project reports │ └── data_processing/ # Contains data processing scripts and reports ├── src/ # Source code directory │ ├── data_processing.py # Script for data cleaning and preprocessing │ ├── feature_engineer.py # Script for feature engineering │ ├── fusion_model.py # Script for model fusion techniques │ ├── model/ # Directory for different models │ ├── model_test.py # Script for testing the model performance ├── .gitignore # Specifies files and directories to be ignored by Git ├── README.md # Project documentation ├── requirements.txt # Python packages and dependencies └── test.ipynb # Jupyter notebook for testing the code


## Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook
- Virtual environment (`venv` or `conda`)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/NN_Assignment2.git

2. Create a virtual environment and activate it:

   ```bash  
    python -m venv .venv
    source .venv/bin/activate  # For Linux/MacOS
    .venv\Scripts\activate     # For Windows
3. Install the required dependencies:
   ```bash  
    pip install -r requirements.txt


