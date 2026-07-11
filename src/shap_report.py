import pandas as pd
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import shap
import matplotlib.pyplot as plt

mlflow.set_tracking_uri('http://localhost:5000')
client = mlflow.MlflowClient()

TARGET = 'prix_xof'
train = pd.read_csv('data/features/train.csv', index_col='date', parse_dates=True)
test = pd.read_csv('data/features/test.csv', index_col='date', parse_dates=True)
drop_cols = [TARGET, 'pays_origine', 'numero_lot']
feature_cols = [c for c in train.columns if c not in drop_cols]
X_te = test[feature_cols]

# --- Charger le modèle champion depuis le Registry ---
champion_version = client.get_latest_versions('cacao_price_predictor', stages=['Staging'])[0]
run = client.get_run(champion_version.run_id)
model_type = run.data.tags.get('model_type')

model_uri = f"models:/cacao_price_predictor/Staging"
if model_type == 'XGBoost':
    model = mlflow.xgboost.load_model(model_uri)
elif model_type == 'LightGBM':
    model = mlflow.lightgbm.load_model(model_uri)
else:
    raise ValueError(
        f"Le champion actuel est un modèle {model_type} (Prophet) : SHAP TreeExplainer "
        "ne s'applique qu'aux modèles à base d'arbres (XGBoost / LightGBM)."
    )

print(f"Modèle champion chargé : {model_type}")

# --- SHAP TreeExplainer (adapté aux modèles à base d'arbres) ---
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_te)

# --- Summary Plot (importance globale des features) ---
plt.figure()
shap.summary_plot(shap_values, X_te, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig('reports/shap_summary.png', dpi=150)
plt.close()
print("Summary plot sauvegardé : reports/shap_summary.png")

# --- Waterfall Plots (3 prédictions locales : début, milieu, fin du test) ---
indices = [0, len(X_te) // 2, len(X_te) - 1]
for i in indices:
    plt.figure()
    shap.plots.waterfall(shap_values[i], show=False)
    plt.tight_layout()
    date_label = X_te.index[i].strftime('%Y-%m-%d')
    plt.savefig(f'reports/shap_waterfall_{date_label}.png', dpi=150)
    plt.close()
    print(f"Waterfall plot sauvegardé pour la date {date_label}")