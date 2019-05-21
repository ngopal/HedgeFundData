import pickle
import os

f = open("./briefings/2016_6_24.p", "rb")
d = pickle.loads(f.read())
print(d)
