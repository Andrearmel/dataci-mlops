import pandas as pd
import sys
import json


def clean(input_path, output_path, metrics_path):
    df = pd.read_csv(input_path, parse_dates=['date'])
    n_before = len(df)

    # 1. Supprimer les doublons sur la colonne date (on garde la 1ère occurrence)
    df = df.drop_duplicates(subset='date', keep='first')

    # 2. Supprimer les lignes avec prix <= 0 (les NaN sont aussi éliminés,
    #    car une comparaison avec NaN renvoie toujours False en pandas)
    df = df[df['prix_xof'] > 0]

    df = df.sort_values('date').reset_index(drop=True)
    n_after = len(df)

    df.to_csv(output_path, index=False)

    metrics = {
        "rows_in": n_before,
        "rows_out": n_after,
        "dropped": n_before - n_after,
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Nettoyage terminé : {metrics}")


if __name__ == '__main__':
    clean(sys.argv[1], sys.argv[2], sys.argv[3])
