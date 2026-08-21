"""
training.py
-----------
Standalone training script for Paddy Yield Predictor.

Steps:
1. Load dataset
2. Clean dataset
3. Check missing values and duplicates
4. Separate features and target
5. Train-test split
6. Build preprocessing pipeline
7. Compare regression models
8. Tune Random Forest Regressor
9. Evaluate final model
10. Calculate Feature Importance
11. Save final model as .joblib
12. Save metrics as CSV

Run:
    python training.py
"""


# ============================================================
# IMPORTS
# ============================================================

import sys
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import (
    RandomizedSearchCV,
    train_test_split
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# training.py is inside src/
# Therefore, project root is one level above src/

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "paddydataset.csv"

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = PROJECT_ROOT / "report"

MODEL_PATH = (
    MODEL_DIR /
    "paddy_yield_predictor.joblib"
)

METRICS_PATH = (
    REPORT_DIR /
    "model_metrics.csv"
)

FEATURE_IMPORTANCE_PATH = (
    REPORT_DIR /
    "feature_importance.csv"
)


# Create required folders
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. CONSTANTS
# ============================================================

TARGET = "Paddy yield(in Kg)"

RANDOM_STATE = 42

TEST_SIZE = 0.20


# ============================================================
# 3. LOGGING
# ============================================================

def log(message):
    """Simple console logger."""

    print(
        f"[INFO] {message}"
    )


def log_error(message):
    """Simple error logger."""

    print(
        f"[ERROR] {message}"
    )


# ============================================================
# 4. LOAD DATA
# ============================================================

def load_dataset():
    """Load Paddy dataset from CSV."""

    log(
        "Starting data loading..."
    )

    log(
        f"Dataset path: {DATA_PATH}"
    )

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found at:\n"
            f"{DATA_PATH}\n\n"
            "Make sure paddydataset.csv "
            "is in the project root."
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

        log(
            "Dataset loaded successfully."
        )

        log(
            f"Dataset shape: {df.shape}"
        )

        return df

    except Exception as e:

        log_error(
            f"Failed to load dataset: {e}"
        )

        raise


# ============================================================
# 5. DATA CLEANING
# ============================================================

def clean_dataset(df):
    """Clean basic dataset issues."""

    log(
        "Starting data cleaning..."
    )

    # Remove completely empty rows
    df = df.dropna(
        axis=0,
        how="all"
    )

    # Remove completely empty columns
    df = df.dropna(
        axis=1,
        how="all"
    )

    # Remove duplicate rows
    duplicate_count = (
        df.duplicated().sum()
    )

    log(
        f"Duplicate rows found: "
        f"{duplicate_count}"
    )

    if duplicate_count > 0:

        df = df.drop_duplicates()

    df = df.reset_index(
        drop=True
    )

    log(
        f"Shape after cleaning: "
        f"{df.shape}"
    )

    return df


# ============================================================
# 6. DATA UNDERSTANDING
# ============================================================

def data_summary(df):
    """Display basic information about dataset."""

    log(
        "Performing data understanding..."
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "PADDY YIELD PREDICTOR"
    )

    print(
        "=" * 60
    )

    print(
        f"\nDataset Shape: {df.shape}"
    )

    print(
        "\nColumn Names:"
    )

    print(
        df.columns.tolist()
    )

    print(
        "\nMissing Values:"
    )

    print(
        df.isnull()
        .sum()
        .sort_values(
            ascending=False
        )
        .head(20)
    )

    print(
        "\nDuplicate Rows:"
    )

    print(
        df.duplicated().sum()
    )

    print(
        "\nData Types:"
    )

    print(
        df.dtypes
    )

    print(
        "\nStatistical Summary:"
    )

    print(
        df.describe().T
    )


# ============================================================
# 7. FEATURE / TARGET SPLIT
# ============================================================

def split_features_target(df):
    """Separate X and y."""

    log(
        "Separating features and target..."
    )

    if TARGET not in df.columns:

        raise KeyError(
            f"Target column "
            f"'{TARGET}' was not found.\n"
            f"Available columns are:\n"
            f"{df.columns.tolist()}"
        )

    # Features
    X = df.drop(
        columns=[TARGET]
    )

    # Target
    y = df[TARGET]

    # Make sure target is numeric
    y = pd.to_numeric(
        y,
        errors="coerce"
    )

    # Remove rows where target is missing
    valid_target = y.notna()

    X = (
        X.loc[valid_target]
        .reset_index(drop=True)
    )

    y = (
        y.loc[valid_target]
        .reset_index(drop=True)
    )

    log(
        f"Features shape: {X.shape}"
    )

    log(
        f"Target shape: {y.shape}"
    )

    print(
        "\nTarget:",
        TARGET
    )

    print(
        "Number of features:",
        X.shape[1]
    )

    print(
        "Number of samples:",
        X.shape[0]
    )

    return X, y


# ============================================================
# 8. TRAIN-TEST SPLIT
# ============================================================

def split_data(X, y):
    """Split dataset into training and testing sets."""

    log(
        "Splitting data into "
        "train and test sets..."
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
    )

    print(
        "\nTraining set:"
    )

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "y_train:",
        y_train.shape
    )

    print(
        "\nTesting set:"
    )

    print(
        "X_test:",
        X_test.shape
    )

    print(
        "y_test:",
        y_test.shape
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# 9. BUILD PREPROCESSOR
# ============================================================

def build_preprocessor(X):
    """Create preprocessing pipeline."""

    log(
        "Building preprocessing pipeline..."
    )

    # Numerical columns
    numeric_features = (
        X.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    # Categorical columns
    categorical_features = (
        X.select_dtypes(
            exclude=["number"]
        )
        .columns
        .tolist()
    )

    print(
        "\nNumerical features:",
        len(numeric_features)
    )

    print(
        "Categorical features:",
        len(categorical_features)
    )

    print(
        "\nCategorical columns:"
    )

    print(
        categorical_features
    )

    # --------------------------------------------------------
    # Numerical pipeline
    # --------------------------------------------------------

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        )
    ])

    # --------------------------------------------------------
    # Categorical pipeline
    # --------------------------------------------------------

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    # --------------------------------------------------------
    # Column Transformer
    # --------------------------------------------------------

    preprocessor = ColumnTransformer([
        (
            "num",
            numeric_pipeline,
            numeric_features
        ),

        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ])

    return preprocessor


# ============================================================
# 10. MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):
    """Calculate regression metrics."""

    predictions = model.predict(
        X_test
    )

    # MAE
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    # MSE
    mse = mean_squared_error(
        y_test,
        predictions
    )

    # RMSE
    rmse = np.sqrt(
        mse
    )

    # R2
    r2 = r2_score(
        y_test,
        predictions
    )

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    }


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

def compare_models(
    X_train,
    X_test,
    y_train,
    y_test,
    X
):
    """Compare multiple regression models."""

    log(
        "Starting model comparison..."
    )

    models = {

        "Linear Regression":
            LinearRegression(),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
                random_state=RANDOM_STATE
            ),

        "Extra Trees":
            ExtraTreesRegressor(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
    }

    results = []

    for name, estimator in models.items():

        try:

            log(
                f"Training {name}..."
            )

            # New preprocessor
            # for every model
            preprocessor = (
                build_preprocessor(X)
            )

            pipeline = Pipeline([
                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "model",
                    estimator
                )
            ])

            pipeline.fit(
                X_train,
                y_train
            )

            metrics = evaluate_model(
                pipeline,
                X_test,
                y_test
            )

            results.append({
                "Model": name,

                "MAE":
                    metrics["MAE"],

                "MSE":
                    metrics["MSE"],

                "RMSE":
                    metrics["RMSE"],

                "R2 Score":
                    metrics["R2 Score"]
            })

            print(
                f"{name} -> "
                f"R2: "
                f"{metrics['R2 Score']:.4f} | "
                f"MAE: "
                f"{metrics['MAE']:.2f} | "
                f"RMSE: "
                f"{metrics['RMSE']:.2f}"
            )

        except Exception as e:

            log_error(
                f"{name} failed: {e}"
            )

    if not results:

        raise RuntimeError(
            "No model could be trained successfully."
        )

    comparison = (
        pd.DataFrame(results)
        .sort_values(
            "R2 Score",
            ascending=False
        )
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "MODEL COMPARISON"
    )

    print(
        "=" * 60
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    return comparison


# ============================================================
# 12. RANDOM FOREST HYPERPARAMETER TUNING
# ============================================================

def tune_random_forest(
    X_train,
    y_train,
    X
):
    """Tune Random Forest using RandomizedSearchCV."""

    log(
        "Starting Random Forest "
        "hyperparameter tuning..."
    )

    # Build preprocessor
    preprocessor = (
        build_preprocessor(X)
    )

    # Random Forest pipeline
    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ])

    # --------------------------------------------------------
    # Hyperparameter grid
    # --------------------------------------------------------

    param_grid = {

        "model__n_estimators": [
            200,
            300,
            500
        ],

        "model__max_depth": [
            None,
            10,
            20,
            30
        ],

        "model__min_samples_split": [
            2,
            5,
            10
        ],

        "model__min_samples_leaf": [
            1,
            2,
            4
        ],

        "model__max_features": [
            "sqrt",
            "log2",
            1.0
        ]
    }

    # --------------------------------------------------------
    # Randomized Search
    # --------------------------------------------------------

    search = RandomizedSearchCV(

        estimator=pipeline,

        param_distributions=param_grid,

        n_iter=15,

        cv=3,

        scoring="neg_root_mean_squared_error",

        random_state=RANDOM_STATE,

        n_jobs=-1,

        verbose=1
    )

    # Train search
    search.fit(
        X_train,
        y_train
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "BEST RANDOM FOREST PARAMETERS"
    )

    print(
        "=" * 60
    )

    print(
        search.best_params_
    )

    print(
        "\nBest CV RMSE:",
        round(
            -search.best_score_,
            2
        )
    )

    return search.best_estimator_


# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(
    model,
    top_n=15
):
    """Calculate and display feature importance."""

    log(
        "Calculating feature importance..."
    )

    # --------------------------------------------------------
    # Get fitted preprocessor
    # --------------------------------------------------------

    preprocessor = (
        model.named_steps[
            "preprocessor"
        ]
    )

    # --------------------------------------------------------
    # Get trained Random Forest
    # --------------------------------------------------------

    rf_model = (
        model.named_steps[
            "model"
        ]
    )

    # --------------------------------------------------------
    # Check feature importance
    # --------------------------------------------------------

    if not hasattr(
        rf_model,
        "feature_importances_"
    ):

        raise AttributeError(
            "The selected model does not "
            "support feature_importances_."
        )

    # --------------------------------------------------------
    # Get importance values
    # --------------------------------------------------------

    importances = (
        rf_model.feature_importances_
    )

    # --------------------------------------------------------
    # Get feature names after preprocessing
    # --------------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    # Safety check
    if len(feature_names) != len(importances):

        raise ValueError(
            "Feature names and importance "
            "values have different lengths."
        )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    importance_df = pd.DataFrame({

        "Feature":
            feature_names,

        "Importance":
            importances
    })

    # --------------------------------------------------------
    # Sort highest to lowest
    # --------------------------------------------------------

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Display Top Features
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "FEATURE IMPORTANCE - RANDOM FOREST"
    )

    print(
        "=" * 60
    )

    print(
        importance_df
        .head(top_n)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save complete feature importance
    # --------------------------------------------------------

    importance_df.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False
    )

    print(
        "\nFeature importance CSV saved:"
    )

    print(
        FEATURE_IMPORTANCE_PATH
    )

    # --------------------------------------------------------
    # Prepare Top Features for Plot
    # --------------------------------------------------------

    top_features = (
        importance_df
        .head(top_n)
        .sort_values(
            by="Importance",
            ascending=True
        )
    )

    # --------------------------------------------------------
    # Plot Feature Importance
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.xlabel(
        "Feature Importance"
    )

    plt.ylabel(
        "Features"
    )

    plt.title(
        "Top 15 Features - "
        "Paddy Yield Prediction"
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Save Plot
    # --------------------------------------------------------

    feature_plot_path = (
        REPORT_DIR /
        "feature_importance.png"
    )

    plt.savefig(
        feature_plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print(
        "\nFeature importance graph saved:"
    )

    print(
        feature_plot_path
    )

    return importance_df


# ============================================================
# 14. SAVE MODEL USING joblib
# ============================================================

def save_model(model):
    """Save trained model pipeline as .joblib."""

    log(
        "Saving final model..."
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"\nModel saved successfully:\n"
        f"{MODEL_PATH}"
    )


# ============================================================
# 15. SAVE METRICS
# ============================================================

def save_metrics(metrics):
    """Save final model metrics to CSV."""

    metrics_df = pd.DataFrame([
        metrics
    ])

    metrics_df.to_csv(
        METRICS_PATH,
        index=False
    )

    print(
        f"Metrics saved successfully:\n"
        f"{METRICS_PATH}"
    )

    return metrics_df


# ============================================================
# 16. SAMPLE PREDICTION
# ============================================================

def test_prediction(
    model,
    X,
    y
):
    """Test final model on first five records."""

    log(
        "Running sample predictions..."
    )

    sample = X.iloc[:5]

    actual = (
        y.iloc[:5]
        .values
    )

    predicted = (
        model.predict(
            sample
        )
    )

    prediction_df = pd.DataFrame({

        "Actual (Kg)":
            actual,

        "Predicted (Kg)":
            predicted,

        "Difference (Kg)":
            np.abs(
                actual -
                predicted
            )
    })

    prediction_df = (
        prediction_df.round(2)
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "SAMPLE PREDICTIONS"
    )

    print(
        "=" * 60
    )

    print(
        prediction_df.to_string(
            index=False
        )
    )


# ============================================================
# 17. MAIN TRAINING FUNCTION
# ============================================================

def main():

    print(
        "\n"
    )

    print(
        "=" * 70
    )

    print(
        "          PADDY YIELD PREDICTOR - MODEL TRAINING"
    )

    print(
        "=" * 70
    )

    try:

        # ----------------------------------------------------
        # Load dataset
        # ----------------------------------------------------

        df = load_dataset()

        # ----------------------------------------------------
        # Clean dataset
        # ----------------------------------------------------

        df = clean_dataset(
            df
        )

        # ----------------------------------------------------
        # Data summary
        # ----------------------------------------------------

        data_summary(
            df
        )

        # ----------------------------------------------------
        # X and y
        # ----------------------------------------------------

        X, y = (
            split_features_target(
                df
            )
        )

        # ----------------------------------------------------
        # Train-test split
        # ----------------------------------------------------

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = split_data(
            X,
            y
        )

        # ----------------------------------------------------
        # Compare models
        # ----------------------------------------------------

        comparison = (
            compare_models(
                X_train,
                X_test,
                y_train,
                y_test,
                X
            )
        )

        # ----------------------------------------------------
        # Save comparison results
        # ----------------------------------------------------

        comparison.to_csv(
            METRICS_PATH,
            index=False
        )

        # ----------------------------------------------------
        # Show best model
        # ----------------------------------------------------

        best_model_name = (
            comparison.iloc[0]["Model"]
        )

        print(
            f"\nBest model before tuning: "
            f"{best_model_name}"
        )

        # ----------------------------------------------------
        # Tune Random Forest
        # ----------------------------------------------------

        best_model = (
            tune_random_forest(
                X_train,
                y_train,
                X
            )
        )

        # ----------------------------------------------------
        # Evaluate tuned Random Forest
        # ----------------------------------------------------

        final_metrics = (
            evaluate_model(
                best_model,
                X_test,
                y_test
            )
        )

        print(
            "\n" + "=" * 60
        )

        print(
            "FINAL RANDOM FOREST RESULTS"
        )

        print(
            "=" * 60
        )

        print(
            f"MAE      : "
            f"{final_metrics['MAE']:.2f} Kg"
        )

        print(
            f"MSE      : "
            f"{final_metrics['MSE']:.2f}"
        )

        print(
            f"RMSE     : "
            f"{final_metrics['RMSE']:.2f} Kg"
        )

        print(
            f"R2 Score : "
            f"{final_metrics['R2 Score']:.4f}"
        )

        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        feature_importance = (
            get_feature_importance(
                best_model,
                top_n=15
            )
        )

        # ----------------------------------------------------
        # Save final metrics
        # ----------------------------------------------------

        save_metrics(
            final_metrics
        )

        # ----------------------------------------------------
        # Save model
        # ----------------------------------------------------

        save_model(
            best_model
        )

        # ----------------------------------------------------
        # Test prediction
        # ----------------------------------------------------

        test_prediction(
            best_model,
            X,
            y
        )

        # ----------------------------------------------------
        # Final message
        # ----------------------------------------------------

        print(
            "\n" + "=" * 70
        )

        print(
            "TRAINING COMPLETED SUCCESSFULLY"
        )

        print(
            "=" * 70
        )

        print(
            f"\nModel file : "
            f"{MODEL_PATH}"
        )

        print(
            f"Metrics    : "
            f"{METRICS_PATH}"
        )

        print(
            f"Feature importance : "
            f"{FEATURE_IMPORTANCE_PATH}"
        )

    except Exception as e:

        log_error(
            f"Training process failed: {e}"
        )

        raise


# ============================================================
# 18. RUN SCRIPT
# ============================================================

if __name__ == "__main__":

    main()