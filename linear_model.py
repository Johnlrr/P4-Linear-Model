from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import pickle
import os
from utils import load_and_clean, feature_engineering_basic, split_data


def train_basic_models(data_path='data/train.csv', out_dir='models'):
    os.makedirs(out_dir, exist_ok=True)
    df = load_and_clean(data_path)
    X, y_o3, y_no2 = feature_engineering_basic(df)
    Xo3_train, Xo3_val, yo3_train, yo3_val = split_data(X, y_o3)
    Xn2_train, Xn2_val, yn2_train, yn2_val = split_data(X, y_no2)

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0, max_iter=10000),
        'SVR Linear': SVR(kernel='linear')
    }
    results = {}
    for name, model in models.items():
        # O3
        model.fit(Xo3_train, yo3_train)
        pred_o3 = model.predict(Xo3_val)
        mae_o3 = mean_absolute_error(yo3_val, pred_o3)
        pickle.dump(model, open(f"{out_dir}/{name}_o3.pkl", 'wb'))
        # NO2
        model.fit(Xn2_train, yn2_train)
        pred_n2 = model.predict(Xn2_val)
        mae_n2 = mean_absolute_error(yn2_val, pred_n2)
        pickle.dump(model, open(f"{out_dir}/{name}_no2.pkl", 'wb'))
        
        results[name] = (mae_o3, mae_n2)
        print(f"{name}: MAE O3={mae_o3:.4f}, NO2={mae_n2:.4f}")
    return results


if __name__ == '__main__':
    train_basic_models()