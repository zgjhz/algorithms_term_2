import os
import pandas as pd

csv_files = [file for file in os.listdir("debug_results") if file.endswith('.csv')]

print(csv_files)

df = pd.DataFrame()

for file in csv_files:
    if "stats" not in file:
        continue
    df_temp = pd.read_csv(os.path.join("debug_results", file))
    df_temp["file"] = file
    df = pd.concat([df, df_temp], ignore_index=True)

df.to_csv("debug_results/joined.csv", index=False)