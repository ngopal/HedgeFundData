import pickle
import os
from keras.preprocessing import *

f = open("./briefings/2016_6_24.p", "rb")
d = pickle.loads(f.read())
print(d)

data = []
for i in os.listdir("./briefings/"):
  print(i)
  f = open("./briefings/"+str(i), "rb")
  d = pickle.loads(f.read())
  data.append(d)

#Headlines
for i,v in enumerate(data):
  for k in v:
    print(k["date"], k["headline"])

# To Build
# N-grams
# Ticker (Entity) Recognition
# Train and Save TF-IDF Vectorizer

#Bodies
#for i,v in enumerate(data):
#  for k in v:
#    print(k["date"], k["text"])

##TickerInfo
