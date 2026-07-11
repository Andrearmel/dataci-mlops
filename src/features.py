import pandas as pd
import numpy as np
import sys


def build_features(input_path, output_path):
    df = pd.read_csv(input_path, parse_dates=['date']).set_index('date').sort_index()

    # --- Features de lag (valeurs passées de prix_xof) ---
    df['lag_1'] = df['prix_xof'].shift(1)     # semaine précédente
    df['lag_4'] = df['prix_xof'].shift(4)     # 4 semaines avant (~1 mois)
    df['lag_30'] = df['prix_xof'].shift(30)   # 30 semaines avant (~7 mois)

    # --- Feature rolling (moyenne mobile sur 8 semaines, sans fuite de données) ---
    df['rolling_mean_8'] = df['prix_xof'].shift(1).rolling(window=8).mean()

    # --- Encodage cyclique du mois (évite la discontinuité déc->jan) ---
    mois = df.index.month
    df['mois_sin'] = np.sin(2 * np.pi * mois / 12)
    df['mois_cos'] = np.cos(2 * np.pi * mois / 12)

    df = df.reset_index()
    df.to_csv(output_path, index=False)
    print(f"Features créées : {df.shape[0]} lignes, {df.shape[1]} colonnes")


if __name__ == '__main__':
    build_features(sys.argv[1], sys.argv[2])
