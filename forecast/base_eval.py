import pandas as pd
import numpy as np

df = pd.read_csv("tsla.csv")
base_eval = np.mean(np.abs(df.iloc[:,1].shift(-1) - df.iloc[:,1]))
print(base_eval)
