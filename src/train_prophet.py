import pandas as pd
import numpy as np
import time
import mlflow
import mlflow.pyfunc
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('cacao_price_prediction')

TARGET = 'prix_xof'
train = pd.read_csv('data/features/train.csv', parse_dates=['date'])
test = pd.read_csv('data/features/test.csv', parse_dates=['date'])

# Prophet impose des colonnes nommées 'ds' (date) et 'y' (valeur à prédire)
train_prophet = train[['date', TARGET]].rename(columns={'date': 'ds', TARGET: 'y'})
test_prophet = test[['date', TARGET]].rename(columns={'date': 'ds', TARGET: 'y'})

# Petit wrapper pour pouvoir logger Prophet comme un modèle MLflow générique (pyfunc)
class ProphetWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        forecast = self.model.predict(model_input[['ds']])
        return forecast['yhat'].values

with mlflow.start_run(run_name='Prophet_v1') as run:
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(train_prophet)

    mlflow.log_param('yearly_seasonality', True)
    mlflow.log_param('weekly_seasonality', False)

    forecast = model.predict(test_prophet[['ds']])
    y_pred = forecast['yhat'].values
    y_test = test_prophet['y'].values

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('mae', mae)
    mlflow.log_metric('r2', r2)

    # Latence sur une seule ligne, 1000 fois
    sample = test_prophet.iloc[[0]][['ds']]
    start = time.time()
    for _ in range(1000):
        model.predict(sample)
    latency_ms = (time.time() - start) / 1000 * 1000

    mlflow.log_metric('latency_ms', latency_ms)
    mlflow.set_tag('model_type', 'Prophet')
    mlflow.set_tag('project', 'DataCI')

    wrapped_model = ProphetWrapper(model)
    mlflow.pyfunc.log_model('model', python_model=wrapped_model)

    print(f"RMSE={rmse:.2f} | MAE={mae:.2f} | R2={r2:.3f} | latence={latency_ms:.1f}ms")