import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def preprocess_data(df: pd.DataFrame):

    target = "Transported"

    X = df.drop(columns=[target])
    y = df[target]

    drop_cols = ["PassengerId", "Name", "Cabin"]
    
    X = X.drop(columns=drop_cols)
    
    categorical_cols = X.select_dtypes(include=["object", "bool"]).columns
    numerical_cols = X.select_dtypes(include=["int64","float64"]).columns
    
    numeric_pipeline = Pipeline([
        ("imputer",
         SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer([("num", numeric_pipeline, numerical_cols), ("cat", categorical_pipeline, categorical_cols)])
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val, preprocessor
    