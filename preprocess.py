import pandas as pd

def prepare_features_for_inference(
    history_df: pd.DataFrame,
    input_df: pd.DataFrame,
    model_features: list
):
    """
    Creates lag_1 and lag_4_mean using Year + Week ordering.
    Assumes:
    - history_df contains Weekly_Sales
    - input_df does NOT contain Weekly_Sales
    """

    history_df = history_df.copy()
    input_df = input_df.copy()

    # ---------- IsHoliday conversion ----------
    for df in [history_df, input_df]:
        df["IsHoliday"] = (
            df["IsHoliday"]
            .astype(str)
            .str.upper()
            .map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0})
            .fillna(0)
            .astype(int)
        )

    # ---------- Mark prediction rows ----------
    input_df["Weekly_Sales"] = pd.NA

    # ---------- Combine history + future ----------
    full_df = pd.concat([history_df, input_df], axis=0)

    # ---------- Sort by time (NO Date) ----------
    full_df = full_df.sort_values(
        by=["Store", "Dept", "Year", "Week"]
    )

    # ---------- Lag features ----------
    full_df["lag_1"] = (
        full_df
        .groupby(["Store", "Dept"])["Weekly_Sales"]
        .shift(1)
    )

    full_df["lag_4_mean"] = (
        full_df
        .groupby(["Store", "Dept"])["Weekly_Sales"]
        .shift(1)
        .rolling(4)
        .mean()
    )

    # ---------- Fill missing lags ----------
    full_df[["lag_1", "lag_4_mean"]] = (
        full_df[["lag_1", "lag_4_mean"]]
        .fillna(0)
    )

    # ---------- Keep only inference rows ----------
    inference_df = full_df[full_df["Weekly_Sales"].isna()].copy()

    # ---------- Feature alignment ----------
    X = inference_df[model_features]

    return X, inference_df
