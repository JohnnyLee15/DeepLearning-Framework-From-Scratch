import pandas as pd

DF_NAME = "housing.csv"
df = pd.read_csv(DF_NAME)
df_clean = df.dropna()
df_clean.to_csv("housing_clean.csv", index=False)

