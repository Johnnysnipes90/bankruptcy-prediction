import streamlit as st
import joblib
import os
import pandas as pd
from config import MODEL_PATH, DATA_DIR, RAW_DATA_PATH  # Import paths

# --- Ensure Dataset Exists Before Loading ---
def load_dataset():
    """Check and load dataset if it exists."""
    if not os.path.exists(RAW_DATA_PATH):
        st.error(f"🚨 Dataset `poland.csv` is missing in `{os.path.dirname(RAW_DATA_PATH)}`.")
        st.stop()

    df = pd.read_csv(RAW_DATA_PATH)

    # Convert 'class' column to a boolean target variable
    if "class" in df.columns:
        df["bankrupt"] = df["class"].astype(bool)
        df.drop(columns=["class"], inplace=True)

    return df


# --- Load Trained Model ---
@st.cache_resource
def load_model():
    """Loads the trained model with validation."""
    with st.spinner("🔄 Loading model..."):
        try:
            with open(MODEL_PATH, "rb") as f:
                model = joblib.load(f)

            if not hasattr(model, "predict"):
                st.error("🚨 Loaded object is not a valid model. Please check `random_forest_model.pkl`.")
                st.stop()

            return model
        except FileNotFoundError:
            st.error(f"🚨 Model file not found at `{MODEL_PATH}`. Ensure the file exists.")
            st.stop()
        except Exception as e:
            st.error(f"🚨 Error loading model: {e}")
            st.stop()


# Load model
model = load_model()


# --- Process Uploaded CSV Data ---
def process_csv(file):
    """Load CSV, handle missing values, and validate format."""
    try:
        df = pd.read_csv(file)

        # Convert 'class' column to boolean and remove it
        if "class" in df.columns:
            df["bankrupt"] = df["class"].astype(bool)
            df.drop(columns=["class"], inplace=True)

        return df
    except Exception as e:
        st.error(f"🚨 Error reading CSV file: {e}")
        return None


# --- Streamlit UI ---
st.title("🏦 Corporate Bankruptcy Prediction")
st.markdown("📊 Predict bankruptcy risk for companies using financial data. Upload a CSV file or select a dataset.")

# --- Sidebar: Dataset Selection ---
st.sidebar.header("📁 Dataset Selection")
option = st.sidebar.radio("Choose data source:", ["Upload File", "Load from Directory"])

df = None  # Placeholder for dataset
selected_file = None  # Track selected filename

if option == "Upload File":
    uploaded_file = st.file_uploader("📤 Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        df = process_csv(uploaded_file)

elif option == "Load from Directory":
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

    if files:
        selected_file = st.selectbox("📂 Select a dataset:", files)
        if selected_file:
            file_path = os.path.join(DATA_DIR, selected_file)
            df = process_csv(file_path)
    else:
        st.warning("⚠ No CSV files found in the directory. Please upload a file.")

# --- Prediction Section ---
if df is not None:
    st.success(f"📊 Using dataset: `{selected_file if option == 'Load from Directory' else uploaded_file.name}`")

    try:
        # Ensure dataset contains expected features
        expected_features = model.feature_names_in_
        missing_features = [col for col in expected_features if col not in df.columns]

        if missing_features:
            st.warning(f"⚠ Missing required features: {missing_features}. Ensure your dataset has all necessary columns.")
        else:
            with st.spinner("🔄 Running bankruptcy prediction..."):
                predictions = model.predict(df[expected_features])

            # Convert to DataFrame
            results = pd.DataFrame({"Bankrupt": predictions})
            results["Bankrupt"] = results["Bankrupt"].astype(bool)

            # Display results
            st.subheader("🔍 Prediction Results")
            st.dataframe(results)

            # Add Download Button
            st.download_button(
                label="📥 Download Predictions",
                data=results.to_csv(index=False),
                file_name="bankruptcy_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"🚨 Error making predictions: {e}")