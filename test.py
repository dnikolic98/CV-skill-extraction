import pandas as pd
import numpy as np

df = pd.read_excel("dataset/training/resumes.xlsx", sheet_name=0)
df = df.replace(np.nan, '', regex=True)

for index, row in df.iterrows():
    print(row[0])
    print("----------------------------------------------------------------------------------")
    print(row[1].strip().split("|"))
    print("==================================================================================")
    
print("Saved model to disk")
