import pandas as pd
import numpy as np
import sys

df = pd.read_csv(sys.argv[1])
base_eval = np.mean(np.abs(df.iloc[:,1].shift(-1) - df.iloc[:,1]))
print(base_eval)
