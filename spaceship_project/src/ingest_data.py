import pandas as pd
import os

def load_data():
    data_path = os.path.join(os.getcwd(), "data")
    
    train_path = os.path.join(data_path, "train_case.csv")
    test_path = os.path.join(data_path, "test_case.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print("Train Data Loaded: ", train_df.shape)
    print("Test Data Loaded: ", test_df.shape)
    
    return train_df, test_df