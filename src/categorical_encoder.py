import pandas as pd

def encode_categorical(df):
    for column in df.select_dtypes(inclide=['object']).columns:
        df[column] = pd.Categorical(df[column]).codes

        return df 