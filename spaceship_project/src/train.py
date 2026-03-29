import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def train_model(X_train, y_train, preprocessor):
    model = LogisticRegression(max_iter=1000)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    pipeline.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    
    pipeline_path = os.path.join("models", "pipeline.pkl")
    
    joblib.dump(pipeline, pipeline_path)
    print("Pipeline Saved To: ", pipeline_path)
    
    return pipeline