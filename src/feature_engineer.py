# src/feature_engineering.py
from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd

def create_interaction_features(df):
    """Create new interaction features from existing features."""
    df['Length_Diameter'] = df['Length'] * df['Diameter']
    return df

def select_features(df, target, k=5):
    """Select top-k important features based on regression or classification task."""
    X = df.drop(columns=[target])
    y = df[target]
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X, y)
    return X.columns[selector.get_support()]
