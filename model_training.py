import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import load_data, preprocess_data, split_data, resample_data  # Updated import
from config import MODEL_PATH  # Updated import

def train_model(X_train, y_train):
    """Train RandomForest model."""
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model and return performance metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def save_model(model):
    """Save trained model."""
    joblib.dump(model, MODEL_PATH)

if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train, y_train = resample_data(X_train, y_train, strategy='over')
    
    model = train_model(X_train, y_train)
    accuracy, report = evaluate_model(model, X_test, y_test)
    
    save_model(model)
    print(f"Model saved! Accuracy: {accuracy:.4f}\n{report}")