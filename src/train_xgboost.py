import pandas as pd
import numpy as np
import time
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('cacao_price_prediction')

TARGET = 'prix_xof'
train = pd.read_csv('data/features/train.csv', index_col='date', parse_dates=True)
test = pd.read_csv('data/features/test.csv', index_col='date', parse_dates=True)

# On enlève les colonnes non numériques / non pertinentes comme features
drop_cols = [TARGET, 'pays_origine', 'numero_lot']
feature_cols = [c for c in train.columns if c not in drop_cols]

X_train, y_train = train[feature_cols], train[TARGET]
X_test, y_test = test[feature_cols], test[TARGET]

params = {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 6,
          'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42}

with mlflow.start_run(run_name='XGBoost_v1'):
    mlflow.log_params(params)

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    # --- Métriques ---
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('mae', mae)
    mlflow.log_metric('r2', r2)

    # --- Latence (1000 prédictions sur une seule ligne, moyenne) ---
    sample = X_test.iloc[[0]]
    start = time.time()
    for _ in range(1000):
        model.predict(sample)
    latency_ms = (time.time() - start) / 1000 * 1000  # ms par prédiction

    mlflow.log_metric('latency_ms', latency_ms)

    # --- Tags utiles pour filtrer/comparer les runs dans l'UI ---
    mlflow.set_tag('model_type', 'XGBoost')
    mlflow.set_tag('project', 'DataCI')

    # --- Modèle (loggué comme artefact MLflow) ---
    mlflow.xgboost.log_model(model, 'model')

    print(f"RMSE={rmse:.2f} | MAE={mae:.2f} | R2={r2:.3f} | latence={latency_ms:.1f}ms")