import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_clean(path: str):
    df = pd.read_csv(path)
    # Xử lý missing hoặc outliers nếu cần
    df = df.dropna()
    return df

def feature_engineering_basic(df):
    # Task 1: chỉ dùng 4 voltage
    X_o3 = df[['o3op1', 'o3op2', 'no2op1', 'no2op2']]
    y_o3 = df['OZONE']
    X_no2 = X_o3.copy()
    y_no2 = df['NO2']
    return X_o3, y_o3, X_no2, y_no2

def feature_engineering_advanced(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['Time'])
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday
    features = ['o3op1','o3op2','no2op1','no2op2','temp','humidity','hour','weekday']
    X = df[features]
    y_o3 = df['OZONE']
    y_no2 = df['NO2']
    return X, y_o3, y_no2

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)