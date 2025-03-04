import pandas as pd
import joblib
import argparse
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from preprocessing import load_data, preprocess_data, split_data, resample_data
from config import MODEL_PATH, PARAM_GRID

def train_model(X_train, y_train):
    """Train RandomForest model with hyperparameter tuning."""
    
    # Define a pipeline with an imputer and Random Forest
    rf_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(random_state=42))
    ])

    # Perform Grid Search with 3-fold cross-validation
    grid_search = GridSearchCV(rf_pipeline, PARAM_GRID, cv=3, scoring="f1", n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)

    # Return the best trained model
    return grid_search.best_estimator_

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["over", "under"], default="over", help="Resampling strategy")
    args = parser.parse_args()

    df = load_data()
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train, y_train = resample_data(X_train, y_train, strategy=args.strategy)

    model = train_model(X_train, y_train)
    accuracy, report = evaluate_model(model, X_test, y_test)

    save_model(model)
    print(f"Model saved! Accuracy: {accuracy:.4f}\n{report}")