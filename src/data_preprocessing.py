import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


TARGET_COLUMN = "loan_status"
CATEGORICAL_COLUMNS = ["education", "self_employed"]


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    return df


def split_features_target(df: pd.DataFrame):
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    # Encode categorical columns
    for col in CATEGORICAL_COLUMNS:
        if col not in X.columns:
            raise ValueError(f"Expected column '{col}' not found in features")

        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])

    # Convert all columns to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Handle missing values
    X = X.fillna(X.median())

    return X
