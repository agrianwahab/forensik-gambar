"""
Utility functions for Forensic Image Analysis System
"""

import numpy as np
from scipy.stats import entropy
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

def detect_outliers_iqr(data, factor=1.5):
    """Detect outliers using IQR method"""
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return np.where((data < lower_bound) | (data > upper_bound))[0]

def calculate_skewness(data):
    """Calculate skewness"""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 3)

def calculate_kurtosis(data):
    """Calculate kurtosis"""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 4) - 3

def normalize_array(arr):
    """Normalize array to 0-1 range"""
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max - arr_min == 0:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)

def safe_divide(numerator, denominator, default=0.0):
    """Safe division with default value"""
    if denominator == 0:
        return default
    return numerator / denominator

HISTORY_FILE = 'analysis_history.json'

def save_analysis_to_history(image_name, analysis_summary, processing_time):
    """Saves the analysis summary to a history file."""
    try:
        history = load_analysis_history()
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'image_name': image_name,
        'analysis_summary': analysis_summary,
        'processing_time': processing_time
    }
    history.append(entry)

    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

def load_analysis_history():
    """Loads the analysis history from a file."""
    try:
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
    except FileNotFoundError:
        print(f"History file '{HISTORY_FILE}' not found. Returning empty list.")
        history = []
    except json.JSONDecodeError:
        print(f"Error decoding JSON from '{HISTORY_FILE}'. Returning empty list.")
        history = []
    return history
