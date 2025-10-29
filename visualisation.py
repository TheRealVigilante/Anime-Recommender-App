import pandas as pd

df = pd.read_csv('data/anime.csv')

# loop through the first row and print each column name and value
for col in df.columns:
    print(f"{col}: {df[col][0]}")
