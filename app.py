import streamlit as st
import pandas as pd
import joblib

from preprocess import prepare_features_for_inference

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Walmart Weekly Sales Forecast",
    layout="wide"
)

st.title("📊 Walmart Weekly Sales Forecasting")
st.write(
    "Upload historical data (with Weekly_Sales) and future data "
    "(same columns except Weekly_Sales) to get predictions."
)

# ---------------- Load Model & Features ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("sales_model.pkl")
    model_features = joblib.load("model_features.pkl")
    return model, model_features

model, model_features = load_artifacts()

# ---------------- File Uploaders ----------------
history_file = st.file_uploader(
    "Upload HISTORICAL data (includes Weekly_Sales)",
    type=["csv"]
)

future_file = st.file_uploader(
    "Upload FUTURE data (excludes Weekly_Sales)",
    type=["csv"]
)

# ---------------- Prediction Logic ----------------
if history_file and future_file:
    try:
        history_df = pd.read_csv(history_file)
        input_df = pd.read_csv(future_file)

        st.subheader("📄 Future Input Preview")
        st.dataframe(input_df.head())

        # ---------- Preprocess & Create Lag Features ----------
        X_test, aligned_df = prepare_features_for_inference(
            history_df=history_df,
            input_df=input_df,
            model_features=model_features
        )

        # ---------- Predict ----------
        predictions = model.predict(X_test)
        predictions = predictions.clip(min=0)

        # ---------- Attach Predictions ----------
        output_df = aligned_df.copy()
        output_df["Predicted_Weekly_Sales"] = predictions

        st.subheader("📈 Prediction Results (Preview)")
        st.dataframe(
            output_df[
                ["Store", "Dept", "Year", "Week", "Predicted_Weekly_Sales"]
            ].head()
        )

        # ---------- Download ----------
        csv = output_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Predictions",
            data=csv,
            file_name="predicted_weekly_sales.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
