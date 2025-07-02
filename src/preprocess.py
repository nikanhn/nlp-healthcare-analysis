import pandas as pd

def preprocess_data(filepath):
    """Loads and preprocesses the healthcare data."""
    df = pd.read_csv(filepath)
    # TODO: Add data cleaning and preprocessing steps
    return df
