import pickle
import os
from keras.preprocessing import *
from nltk.stem import SnowballStemmer

f = open("./briefings/2016_6_24.p", "rb")
d = pickle.loads(f.read())
print(d)

data = []
for i in os.listdir("./briefings/"):
  #print(i)
  f = open("./briefings/"+str(i), "rb")
  d = pickle.loads(f.read())
  data.append(d)

#Headlines
headlines = []
text_bodies = []
all_text = []
snow = SnowballStemmer('english')
for i,v in enumerate(data):
  for k in v:
    #print(k["date"], k["headline"])
    headlines.append(' '.join([snow.stem(w) for w in str(k["headline"])]))
    text_bodies.append(' '.join([snow.stem(w) for w in str(k["text"])]))
    all_text.append(' '.join([snow.stem(w) for w in str(k["headline"])]))
    all_text.append(' '.join([snow.stem(w) for w in str(k["text"])]))

tokenizer = text.Tokenizer(num_words=100, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', char_level=False, oov_token=None)

tokenizer.fit_on_texts(all_text)
encoded_docs = tokenizer.texts_to_matrix(all_text, mode='tfidf')
print(encoded_docs)
print(tokenizer.word_index[0:10])

# To Build
# N-grams
# Ticker (Entity) Recognition
# Train and Save TF-IDF Vectorizer

#Bodies
#for i,v in enumerate(data):
#  for k in v:
#    print(k["date"], k["text"])

##TickerInfo