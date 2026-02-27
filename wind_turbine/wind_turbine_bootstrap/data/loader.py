import numpy as np
import pandas as pd

def load_data(file_path):
    """Load raw CSV data into numpy array."""
    return np.array(pd.read_csv(file_path, header=None))

def preprocess_data(X_raw):
    """Invert data (avoid zeros)."""
    X_raw = X_raw[:, ~np.isnan(X_raw).any(axis=0)]
    return 1/np.array(X_raw)

