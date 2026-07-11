import pandas as pd
import numpy as np
import time
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('cacao_price_prediction')

TARGET = 'prix_xof'
train = pd.read_csv('data/features/train.csv', index_col='date', parse_dates=True)
test = pd.read_csv('data/features/test.csv', index_col='date', parse_dates=True)

drop_cols = [TARGET, 'pays_origine', 'numero_lot']
feature_cols = [c for c in train.columns if c not in drop_cols]

X_train, y_train = train[feature_cols], train[TARGET]
X_test, y_test = test[feature_cols], test[TARGET]

params_lgb = {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 6,
              'num_leaves': 63, 'random_state': 42, 'verbose': -1}

with mlflow.start_run(run_name='LightGBM_v1'):
    mlflow.log_params(params_lgb)

    model = lgb.LGBMRegressor(**params_lgb)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('mae', mae)
    mlflow.log_metric('r2', r2)

    sample = X_test.iloc[[0]]
    start = time.time()
    for _ in range(1000):
        model.predict(sample)
    latency_ms = (time.time() - start) / 1000 * 1000

    mlflow.log_metric('latency_ms', latency_ms)
    mlflow.set_tag('model_type', 'LightGBM')
    mlflow.set_tag('project', 'DataCI')

    mlflow.lightgbm.log_model(model, 'model')

    print(f"RMSE={rmse:.2f} | MAE={mae:.2f} | R2={r2:.3f} | latence={latency_ms:.1f}ms")