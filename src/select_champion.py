import mlflow

mlflow.set_tracking_uri('http://localhost:5000')

# Récupère tous les runs de l'expérience, triés par RMSE croissant
runs = mlflow.search_runs(
    experiment_names=['cacao_price_prediction'],
    order_by=['metrics.rmse ASC']
)

print(runs[['tags.mlflow.runName', 'metrics.rmse', 'metrics.latency_ms']].to_string())

# --- Grille de sélection ---
# 1. RMSE < 130 XOF   2. Latence < 50ms   3. Taille < 50 MB   4. Rapport SHAP dispo
RMSE_MAX = 130
LATENCY_MAX = 50

eligible = runs[(runs['metrics.rmse'] < RMSE_MAX) & (runs['metrics.latency_ms'] < LATENCY_MAX)]

if len(eligible) == 0:
    print("\nAucun modèle ne respecte les 2 premiers critères (RMSE<130 ET latence<50ms).")
    print("On sélectionne le meilleur compromis disponible (RMSE minimal parmi les runs sous la limite de latence).")
    eligible = runs[runs['metrics.latency_ms'] < LATENCY_MAX]

champion_run = eligible.iloc[0]
champion_run_id = champion_run['run_id']
champion_name = champion_run['tags.mlflow.runName']

print(f"\nChampion sélectionné : {champion_name} (run_id={champion_run_id})")
print(f"RMSE={champion_run['metrics.rmse']:.2f} | latence={champion_run['metrics.latency_ms']:.1f}ms")

# --- Enregistrement dans le Model Registry ---
model_uri = f"runs:/{champion_run_id}/model"
result = mlflow.register_model(model_uri=model_uri, name="cacao_price_predictor")

# --- Promotion en Staging ---
client = mlflow.MlflowClient()
client.transition_model_version_stage(
    name="cacao_price_predictor",
    version=result.version,
    stage="Staging"
)

print(f"\nChampion en Staging : version {result.version}")