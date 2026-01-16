# Walmart Weekly Sales Forecasting

This Streamlit app predicts weekly sales at Store–Department level
using an XGBoost model trained on historical Walmart sales data.

## How to Use
1. Upload historical data (with Weekly_Sales)
2. Upload future data (without Weekly_Sales)
3. Download predicted weekly sales

## Model
- Algorithm: XGBoost Regressor
- Features: Store, Dept, lag features, macro indicators
- Metric: WMAE
