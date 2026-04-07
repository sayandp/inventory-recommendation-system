# src/models/split.py

def split_data(df, split_date="2011-10-01"):

    train = df[df["date"] < split_date]
    test = df[df["date"] >= split_date]

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    return train, test