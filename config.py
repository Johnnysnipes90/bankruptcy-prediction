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
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")  # Updated model name
RAW_DATA_PATH = os.path.join(DATA_DIR, "poland.csv")  # Updated dataset name
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_data.csv")

# Hyperparameter settings for GridSearchCV
PARAM_GRID = {
    "rf__n_estimators": [150, 200, 250],  
    "rf__max_depth": [10, 15, 20],        
    "rf__min_samples_split": [2, 5],      
    "rf__min_samples_leaf": [1, 2],       
    "rf__class_weight": ["balanced"]
}

# FastAPI settings
API_HOST = "127.0.0.1"
API_PORT = 8000