import streamlit as st
import pandas as pd
import joblib

from preprocess import prepare_features_for_inference

# ---------------- Sample Input ----------------
def get_sample_input_df():
    sample_data = {
        "Store": [1],
        "Dept": [1],
        "IsHoliday": [0],
        "Temperature": [55.3],
        "Fuel_Price": [3.38],
        "MarkDown1": [6766.44],
        "MarkDown2": [5147.7],
        "MarkDown3": [50.82],
        "MarkDown4": [3639.9],
        "MarkDown5": [2737.42],
        "CPI": [223.46],
        "Unemployment": [6.57],
        "Type": [0],
        "Size": [151315],
        "Year": [2012],
        "Month": [11],
        "Week": [44],
        "day": [2]
    }
    return pd.DataFrame(sample_data)

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

# ---------------- Sample Table ----------------
st.subheader("📄 Sample Input Format")

with st.expander("📘 Dataset Column Descriptions"):
    st.markdown(
        """
###  Store & Time Identifiers
- **Store**: Unique ID for each Walmart store (1–45)
- **Dept**: Department number within the store
- **Year**: Calendar year of the sales week
- **Month**: Month of the year (1–12)
- **Week**: Week number of the year
- **day**: Day of the month when the week starts

###  Target Variable
- **Weekly_Sales**: Total weekly sales for a department *(used only in historical data)*

###  Holiday Indicator
- **IsHoliday**:  
  - `1` → Holiday week  
  - `0` → Non-holiday week  

Major holidays:
- Super Bowl  
- Labor Day  
- Thanksgiving  
- Christmas  

###  Economic & Environmental Features
- **Temperature**: Average weekly regional temperature
- **Fuel_Price**: Average fuel price in the region
- **CPI**: Consumer Price Index
- **Unemployment**: Regional unemployment rate

###  Promotional Features
- **MarkDown1 – MarkDown5**:  
  Anonymized promotional markdown indicators  
  - Mostly available after Nov 2011  
  - Missing values indicate no promotion

###  Store Characteristics
- **Type**: Store type (encoded)
- **Size**: Physical store size (square feet)

###  Engineered Lag Features ( will be calculated by the model)
- **lag_1**: Sales from the previous week (same Store & Dept)
- **lag_4_mean**: Average sales of the previous 4 weeks (excluding current week)
"""
    )
    
st.markdown(
    """
    **Required columns for upload**
    - Same columns as training data
    - `Weekly_Sales` must be present in historical data
    - `Weekly_Sales` must NOT be present in future data
    """
)

sample_df = get_sample_input_df()
st.dataframe(sample_df, use_container_width=True)

# ---------------- Load Model ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("sales_model.pkl")
    model_features = joblib.load("model_features.pkl")
    return model, model_features

model, model_features = load_artifacts()

# ---------------- File Upload ----------------
history_file = st.file_uploader(
    "Upload HISTORICAL data (includes Weekly_Sales)",
    type=["csv"]
)

future_file = st.file_uploader(
    "Upload FUTURE data (excludes Weekly_Sales)",
    type=["csv"]
)

# ---------------- Prediction ----------------
if history_file and future_file:
    try:
        history_df = pd.read_csv(history_file)
        input_df = pd.read_csv(future_file)

        st.subheader("📄 Future Data Preview")
        st.dataframe(input_df.head())

        X_test, aligned_df = prepare_features_for_inference(
            history_df=history_df,
            input_df=input_df,
            model_features=model_features
        )

        predictions = model.predict(X_test)
        predictions = predictions.clip(min=0)

        output_df = aligned_df.copy()
        output_df["Predicted_Weekly_Sales"] = predictions

        st.subheader("📈 Predictions Preview")
        st.dataframe(
            output_df[["Store", "Dept", "Year", "Week", "Predicted_Weekly_Sales"]].head()
        )

        csv = output_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download Predictions",
            csv,
            "predicted_weekly_sales.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")




