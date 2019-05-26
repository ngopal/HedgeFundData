import pickle
import os

f = open("./cboe/4_1_2016_cboe_futures.dat", "rb")
d = pickle.loads(f.read())
print(d)
