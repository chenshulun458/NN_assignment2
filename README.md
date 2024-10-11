# Assignment2_group72
abalone-age-prediction/
│
├── data/                               # Directory to store all data-related files
│   ├── raw/                            # Directory for storing raw, unprocessed data files
│   ├── processed/                      # Directory for storing preprocessed data files
│   └── README.md                       # Documentation of the data directory and files
│
├── notebooks/                          # Jupyter notebooks directory
│   ├── exploratory_analysis.ipynb       # Notebook for exploratory data analysis (EDA)
│   └── model_comparison.ipynb           # Notebook for model comparison and results visualization
│
├── src/                                # Source code directory
│   ├── __init__.py                     # Initialization file for making 'src' a package
│   ├── config/                         # Configuration directory
│   │   └── config.yaml                 # Configuration file (e.g., hyperparameters, file paths)
│   ├── data_preprocessing.py           # Module for data cleaning and preprocessing functions
│   ├── feature_engineering.py          # Module for feature engineering and feature selection
│   ├── model.py                        # Module for defining and training models
│   ├── evaluation.py                   # Module for evaluating models and computing metrics
│   ├── model_fusion.py                 # Module for model fusion and ensemble techniques
│   └── utils.py                        # Utility functions (e.g., data loading, visualization)
│
├── experiments/                        # Directory for storing experiment results
│   ├── baseline/                       # Directory for baseline model results
│   ├── regression/                     # Directory for regression task results
│   ├── classification/                 # Directory for classification task results
│   └── fusion/                         # Directory for storing model fusion results
│
├── reports/                            # Directory for project reports
│   ├── figures/                        # Directory for storing figures and visualizations
│   ├── abalone_report.pdf              # Final project report in PDF format
│── README.md                       # Documentation for the reports directory
├── requirements.txt                    # List of dependencies for Python packages
