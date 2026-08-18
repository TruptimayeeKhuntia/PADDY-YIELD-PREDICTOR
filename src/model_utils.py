"""
model_utils.py
--------------
Builds the preprocessing pipeline and the final model pipeline.
"""

import numpy as np
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.logger import get_logger

log = get_logger(__name__)


def build_preprocessor(X):
    """
    Builds a ColumnTransformer that:
    - For numeric columns: fills missing values with median, then scales
    - For categorical columns: fills missing with most frequent, then one-hot encodes
    """
    try:
        num_cols = X.select_dtypes(include="number").columns.tolist()
        cat_cols = X.select_dtypes(exclude="number").columns.tolist()

        log.info(f"Numeric features: {len(num_cols)} | Categorical features: {len(cat_cols)}")

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ])

        return preprocessor

    except Exception as e:
        log.error(f"Error building preprocessor: {e}")
        raise


def build_model(X, n_estimators=300):
    """
    Builds the full pipeline: preprocessor + RandomForest model.
    """
    try:
        preprocessor = build_preprocessor(X)

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1
            ))
        ])

        log.info(f"Model pipeline built with n_estimators={n_estimators}")
        return model

    except Exception as e:
        log.error(f"Error building model: {e}")
        raise


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluates the trained model and returns a dict of metrics.
    """
    try:
        preds = model.predict(X_test)

        metrics = {
            "MAE":      round(mean_absolute_error(y_test, preds), 2),
            "RMSE":     round(np.sqrt(mean_squared_error(y_test, preds)), 2),
            "R2 Score": round(r2_score(y_test, preds), 5)
        }

        log.info(f"Evaluation — MAE: {metrics['MAE']} | RMSE: {metrics['RMSE']} | R²: {metrics['R2 Score']}")
        return metrics

    except Exception as e:
        log.error(f"Error during model evaluation: {e}")
        raise


def save_model(model, path):
    """
    Saves the trained model to disk using joblib.
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        log.info(f"Model saved to: {path}")

    except Exception as e:
        log.error(f"Failed to save model: {e}")
        raise


def load_model(path):
    """
    Loads a saved model from disk.
    """
    try:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        model = joblib.load(path)
        log.info(f"Model loaded from: {path}")
        return model

    except FileNotFoundError as e:
        log.error(str(e))
        raise

    except Exception as e:
        log.error(f"Failed to load model: {e}")
        raise
