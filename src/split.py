import pandas as pd
import sys


def split(input_path, train_out, test_out):
    df = pd.read_csv(input_path, index_col='date', parse_dates=True).sort_index()

    # Split temporel : 80% train / 20% test — jamais aléatoire !
    # (un split aléatoire mélangerait passé et futur -> fuite d'information)
    n = len(df)
    cut = int(n * 0.8)

    train = df.iloc[:cut]
    test = df.iloc[cut:]

    train.to_csv(train_out)
    test.to_csv(test_out)

    print(f"Train : {train.shape} ({train.index.min()} -> {train.index.max()})")
    print(f"Test  : {test.shape} ({test.index.min()} -> {test.index.max()})")


if __name__ == '__main__':
    split(sys.argv[1], sys.argv[2], sys.argv[3])
