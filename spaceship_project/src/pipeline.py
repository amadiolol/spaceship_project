from src.ingest_data import load_data
from src.preprocess import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model

def run_pipeline():
    train_df, test_df = load_data()
    
    X_train, X_val, y_train, y_val, preprocessor = preprocess_data(train_df)
    
    model = train_model(X_train, y_train, preprocessor)
    
    accuracy = evaluate_model(model, X_val, y_val)
    
    print("Final Validation Accuracy: ", accuracy)
    
if __name__ == "__main__":
    run_pipeline()