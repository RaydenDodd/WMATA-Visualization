import pandas as pd

df = pd.read_csv("Entries_by_Year_Full_Data_data.csv")
df_sample = df.sample(n=500_000, random_state=42)
df_sample.to_csv("small_sample.csv", index=False)
