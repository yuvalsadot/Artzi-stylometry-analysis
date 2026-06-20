import pandas as pd

df = pd.read_excel("../songs/songs_summary.xlsx")

print(df.head())
print()
print(df.columns)