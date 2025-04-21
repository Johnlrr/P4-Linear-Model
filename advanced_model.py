import os, pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from utils import load_and_clean, feature_engineering_advanced, split_data


def train_advanced_models(data_path='data/train.csv', out_dir='models'):
    os.makedirs(out_dir, exist_ok=True)
    df = load_and_clean(data_path)
    X, y_o3, y_no2 = feature_engineering_advanced(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pickle.dump(scaler, open(f"{out_dir}/scaler_advanced.pkl", 'wb'))

    best_models = {}
    grids = {
        'Random Forest': {'model': RandomForestRegressor(random_state=42),
               'params': {'n_estimators': [50,100], 'max_depth': [None,10,20], 'min_samples_split': [2,5]}},
        'Gradient Boosting': {'model': GradientBoostingRegressor(random_state=42),
               'params': {'n_estimators': [50,100], 'learning_rate': [0.1,0.01], 'max_depth': [3,5]}},
        'Decision Tree': {'model': DecisionTreeRegressor(random_state=42),
               'params': {'max_depth': [None,5,10], 'min_samples_split': [2,5,10]}},
        'K-Nearest Neighbors': {'model': KNeighborsRegressor(),
                'params': {'n_neighbors': [3,5,7], 'weights': ['uniform','distance']}}
    }

    for target, y in [('o3', y_o3), ('no2', y_no2)]:
        X_train, X_val, y_train, y_val = split_data(X_scaled, y)
        best_mae = float('inf')
        best_model = None
        for name, cfg in grids.items():
            gs = GridSearchCV(cfg['model'], cfg['params'], cv=5,
                              scoring='neg_mean_absolute_error', n_jobs=-1)
            gs.fit(X_train, y_train)
            mae = mean_absolute_error(y_val, gs.best_estimator_.predict(X_val))
            print(f"{name.upper()} {target}: MAE={mae:.4f}")
            if mae < best_mae:
                best_mae = mae
                best_model = gs.best_estimator_

        filename = f"{out_dir}/advanced_best_model_{target}.pkl"
        pickle.dump(best_model, open(filename, 'wb'))
        print(f"Saved best {target} model to {filename} with MAE={best_mae:.4f}\n")
        best_models[target] = best_model
    return best_models

if __name__ == '__main__':
    train_advanced_models()