import os

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths for data and models
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Model and data file paths
MODEL_PATH = os.path.join(MODEL_DIR, "best_rf_model.pkl")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_data.arff")  # Updated path for consistency
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_data.csv")  # Ensure processed data is correctly referenced

# Hyperparameter settings for GridSearchCV
PARAM_GRID = {
    "n_estimators": [150, 200, 250],  # Removed "rf__" prefix since it's a direct model
    "max_depth": [10, 15, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "class_weight": ["balanced"]
}

# FastAPI settings
API_HOST = "127.0.0.1"
API_PORT = 8000