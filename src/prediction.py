"""
prediction.py
-------------
Prediction script for Paddy Yield Predictor.

Steps:
1. Load trained .joblib model
2. Load dataset
3. Separate features and target
4. Make predictions
5. Compare actual and predicted yield
6. Display prediction results

Run:
    python src/prediction.py
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import joblib
import pandas as pd
import numpy as np


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# prediction.py is inside src/
# Therefore, project root is one level above src/

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

# Dataset is in project root
DATA_PATH = (
    PROJECT_ROOT /
    "paddydataset.csv"
)

# Trained model is inside models folder
MODEL_PATH = (
    PROJECT_ROOT /
    "models" /
    "paddy_yield_predictor.joblib"
)


# ============================================================
# 2. CONSTANTS
# ============================================================

TARGET = "Paddy yield(in Kg)"


# ============================================================
# 3. LOAD MODEL
# ============================================================

def load_model():
    """Load the trained Paddy Yield Predictor model."""

    print(
        "[INFO] Loading trained model..."
    )

    print(
        f"[INFO] Model path: {MODEL_PATH}"
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found at:\n"
            f"{MODEL_PATH}\n\n"
            "Run training.py first."
        )

    try:

        model = joblib.load(
            MODEL_PATH
        )

        print(
            "[INFO] Model loaded successfully."
        )

        return model

    except Exception as e:

        print(
            f"[ERROR] Failed to load model: {e}"
        )

        raise


# ============================================================
# 4. LOAD DATASET
# ============================================================

def load_dataset():
    """Load dataset for testing predictions."""

    print(
        "\n[INFO] Loading dataset..."
    )

    print(
        f"[INFO] Dataset path: {DATA_PATH}"
    )

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found at:\n"
            f"{DATA_PATH}"
        )

    try:

        df = pd.read_csv(
            DATA_PATH
        )

        # Clean column names
        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
        )

        # Remove duplicate rows
        df = (
            df
            .drop_duplicates()
            .reset_index(drop=True)
        )

        print(
            "[INFO] Dataset loaded successfully."
        )

        print(
            f"[INFO] Dataset shape: {df.shape}"
        )

        return df

    except Exception as e:

        print(
            f"[ERROR] Failed to load dataset: {e}"
        )

        raise


# ============================================================
# 5. PREPARE DATA
# ============================================================

def prepare_data(df):
    """Separate features and target."""

    print(
        "\n[INFO] Preparing data..."
    )

    if TARGET not in df.columns:

        raise KeyError(
            f"Target column '{TARGET}' "
            "not found."
        )

    # Separate features
    X = df.drop(
        columns=[TARGET]
    )

    # Separate target
    y = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    # Remove rows with missing target
    valid_rows = y.notna()

    X = (
        X.loc[valid_rows]
        .reset_index(drop=True)
    )

    y = (
        y.loc[valid_rows]
        .reset_index(drop=True)
    )

    print(
        f"[INFO] Features shape: {X.shape}"
    )

    print(
        f"[INFO] Target shape: {y.shape}"
    )

    return X, y


# ============================================================
# 6. MAKE PREDICTIONS
# ============================================================

def make_predictions(
    model,
    X,
    y,
    number_of_samples=5
):
    """Predict yield for selected dataset rows."""

    number_of_samples = min(
        number_of_samples,
        len(X)
    )

    if number_of_samples == 0:

        raise ValueError(
            "No valid samples available "
            "for prediction."
        )

    # Select samples
    X_sample = X.iloc[
        :number_of_samples
    ]

    # Actual values
    y_actual = (
        y.iloc[
            :number_of_samples
        ].values
    )

    print(
        f"\n[INFO] Predicting "
        f"{number_of_samples} samples..."
    )

    # Make predictions
    predictions = model.predict(
        X_sample
    )

    # Create result DataFrame
    prediction_df = pd.DataFrame({

        "Actual Yield (Kg)":
            y_actual,

        "Predicted Yield (Kg)":
            predictions,

        "Difference (Kg)":
            np.abs(
                y_actual -
                predictions
            )
    })

    # Round values
    prediction_df = (
        prediction_df.round(2)
    )

    return prediction_df


# ============================================================
# 7. DISPLAY RESULT
# ============================================================

def display_results(
    prediction_df
):
    """Display prediction results."""

    print(
        "\n"
    )

    print(
        "=" * 75
    )

    print(
        "             PADDY YIELD PREDICTION RESULTS"
    )

    print(
        "=" * 75
    )

    print(
        prediction_df.to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 75
    )


# ============================================================
# 8. MAIN
# ============================================================

def main():

    print(
        "\n"
    )

    print(
        "=" * 75
    )

    print(
        "              PADDY YIELD PREDICTOR"
    )

    print(
        "                    PREDICTION"
    )

    print(
        "=" * 75
    )

    try:

        # ----------------------------------------------------
        # Load trained model
        # ----------------------------------------------------

        model = load_model()

        # ----------------------------------------------------
        # Load dataset
        # ----------------------------------------------------

        df = load_dataset()

        # ----------------------------------------------------
        # Prepare X and y
        # ----------------------------------------------------

        X, y = prepare_data(
            df
        )

        # ----------------------------------------------------
        # Make predictions
        # ----------------------------------------------------

        prediction_df = (
            make_predictions(
                model,
                X,
                y,
                number_of_samples=5
            )
        )

        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        display_results(
            prediction_df
        )

        print(
            "\n[INFO] Prediction completed successfully."
        )

    except Exception as e:

        print(
            f"\n[ERROR] Prediction failed: {e}"
        )

        raise


# ============================================================
# 9. RUN
# ============================================================

if __name__ == "__main__":

    main()