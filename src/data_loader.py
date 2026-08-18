"""
data_loader.py
--------------
Handles loading and basic cleaning of the paddy dataset.
"""

from pathlib import Path
import pandas as pd
from src.logger import get_logger

log = get_logger(__name__)

TARGET = "Paddy yield(in Kg)"


def load_data(data_path=None) -> pd.DataFrame:
    """
    Load the CSV dataset.
    If no path is given, it looks for the file in the project root.
    """
    try:
        if data_path is None:
            data_path = Path(__file__).resolve().parents[1] / "paddydataset(2).csv"

        data_path = Path(data_path)

        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {data_path}")

        df = pd.read_csv(data_path)
        df.columns = df.columns.astype(str).str.strip()

        log.info(f"Dataset loaded — shape: {df.shape}")
        return df

    except FileNotFoundError as e:
        log.error(f"File not found: {e}")
        raise

    except Exception as e:
        log.error(f"Unexpected error while loading data: {e}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Remove rows/columns that are completely empty
    - Remove duplicate rows
    - Reset the index
    """
    try:
        before = df.shape[0]

        df = (
            df
            .dropna(axis=0, how="all")
            .dropna(axis=1, how="all")
            .drop_duplicates()
            .reset_index(drop=True)
        )

        removed = before - df.shape[0]
        log.info(f"Cleaning done — removed {removed} duplicate/empty rows. Final shape: {df.shape}")
        return df

    except Exception as e:
        log.error(f"Error during data cleaning: {e}")
        raise


def split_features_target(df: pd.DataFrame):
    """
    Splits the dataframe into features (X) and target (y).
    Returns X, y as separate DataFrames.
    """
    try:
        if TARGET not in df.columns:
            raise ValueError(f"Target column '{TARGET}' not found in dataset.")

        X = df.drop(columns=[TARGET])
        y = df[TARGET]

        log.info(f"Features shape: {X.shape} | Target shape: {y.shape}")
        return X, y

    except ValueError as e:
        log.error(f"Column error: {e}")
        raise

    except Exception as e:
        log.error(f"Unexpected error in split: {e}")
        raise
