import pandas as pd
import pickle
import os

TEST_PATH = 'data/dummy_test.csv'
O3_MODEL = 'models/advanced_best_model_o3.pkl'
NO2_MODEL = 'models/advanced_best_model_no2.pkl'
SCALER   = 'models/scaler_advanced.pkl'
OUTPUT_PATH = 'predictions.csv'

if __name__ == '__main__':
    df = pd.read_csv(TEST_PATH)

    df['Time'] = pd.to_datetime(df['Time'])
    df['hour'] = df['Time'].dt.hour
    df['weekday'] = df['Time'].dt.weekday
    features = ['o3op1','o3op2','no2op1','no2op2','temp','humidity','hour','weekday']
    X = df[features]

    # Load scaler and scale
    scaler = pickle.load(open(SCALER, 'rb'))
    X_scaled = scaler.transform(X)

    # Load models and predict
    model_o3 = pickle.load(open(O3_MODEL, 'rb'))
    model_no2 = pickle.load(open(NO2_MODEL, 'rb'))
    preds_o3 = model_o3.predict(X_scaled)
    preds_no2 = model_no2.predict(X_scaled)

    # Save predictions to CSV
    out_df = pd.DataFrame({
        'O3_pred': preds_o3,
        'NO2_pred': preds_no2
    })
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Predictions saved to {OUTPUT_PATH}")