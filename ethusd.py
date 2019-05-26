import pickle
import os, sys
import pandas as pd
import keras
import pandas as pd
import numpy as np
from scipy import signal
from keras.layers import *
from keras.activations import *
from keras.callbacks import *
from keras.optimizers import *
from keras.models import *
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

# Read Files
dd = defaultdict(list)
for i in os.listdir("./data/files/eth/"):
  d = pickle.load(open("./data/files/eth/"+i, "rb"))
  for e in d:
    dd[e[0]] = e[1]

# Get Data into DataFrame
data_rows = []
for k in sorted(dd.items(), key=lambda x: x[0], reverse=True):
  data_rows.append(k[1])

df = pd.DataFrame(data_rows)
df.rename({0:"time", 1:"low", 2:"high", 3:"open", 4:"close", 5:"volume"}, inplace=True, axis=1)

# Find Crests and Troughs
shifted_df = df.iloc[:,1:].shift(-1)

combined_df = df.merge(shifted_df, left_index=True, right_index=True, suffixes=('_left', '_right')).dropna()

# De-trend data
data_orig_detrended = signal.detrend(combined_df.iloc[:,1:])

scaler = MinMaxScaler(feature_range=(0, 1))
data_mat = scaler.fit_transform(data_orig_detrended)

print(data_mat)
# 'time', 'low_left', 'high_left', 'open_left', 'close_left', 'volume_left',
#       'low_right', 'high_right', 'open_right', 'close_right', 'volume_right'

seq_len = 10 # days to use for prediction
data = np.array((data_mat))

sequence_length = seq_len + 1
result = []
for index in range(len(data) - sequence_length):
    result.append(data[index: index + sequence_length])

result = np.array(result)

row = round(0.9 * result.shape[0])
train = result[:int(row), :] # Create training set

x_train = result[:int(row), 0:5]
y_train = data_mat[:int(row), 5:]
x_test = result[int(row):, 0:5]
y_test = data_mat[int(row), 5:]

print("TrainX", x_train)
print("TrainY", y_train)
print("TestX", x_test)
print("TestY", y_test)


LAYERS = 120
model = Sequential()

model.add(LSTM(
    input_dim=data.shape[1],
    output_dim=LAYERS,
    return_sequences=True))
model.add(BatchNormalization())
model.add(LSTM(
    LAYERS,
    return_sequences=False))
model.add(Dropout(0.5))

model.add(Dense(
    output_dim=y_train.shape[1]))
model.add(Activation('linear'))

start = time.time()
rmsprop = RMSprop(lr=0.1)
model.compile(loss='mse', optimizer='adam')
print('compilation time : ', time.time() - start)

MODELNAME = 'simple_ethusd'
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('./forecast/models/'+MODELNAME+'_best.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, epsilon=1e-4, mode='min')


VALIDATIONSIZE = 0.25
EPOCHS = 500
#model = keras.models.load_model('./forecast/models/'+MODELNAME+'_best.hdf5')
history = model.fit(
    x_train,
    y_train,
    batch_size=512,
    nb_epoch=EPOCHS,
    validation_split=VALIDATIONSIZE,
    callbacks = [reduce_lr_loss, earlyStopping, mcp_save],
    shuffle=True)

best_model = keras.models.load_model('./forecast/models/'+MODELNAME+'_best.hdf5')
days=10
print(best_model.predict(np.reshape(data[-days:], (days, 1, data.shape[1]))))

plt.plot(best_model.predict(np.reshape(data[-days:], (days, 1, data.shape[1])))[2])
plt.savefig("./superpic.png")
