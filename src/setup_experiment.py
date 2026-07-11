import mlflow

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('cacao_price_prediction')
print("Expérience créée avec succès")