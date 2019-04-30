import sys, os
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import load_model
import lstm, time #helper libraries
import numpy as np
import pandas as pd

INPUTTICKER = sys.argv[1] or "QQQ"
print(INPUTTICKER)
FILENAME = 'multiple_concatenated_tickers'
MODELNAME = FILENAME.split('.')[0]
TIMEHORIZON = 5
EPOCHS = 500
LAYERS = 50
VALIDATIONSIZE = 0.20
normalise_window = False

time_horizon = TIMEHORIZON
data = pd.read_csv('./data/files/multiple_concatenated_tickers.csv')
data_mat = data.iloc[:,1:].as_matrix()

ticker_dict = dict([(i[1].split('.')[0], int(i[0])-1) for i in enumerate(list(data.columns)) if 'Open' in i[1]])

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

seq_len = time_horizon
data = np.array(normalise_windows(data_mat))

sequence_length = seq_len + 1
result = []
for index in range(len(data) - sequence_length):
    result.append(data[index: index + sequence_length])

if normalise_window:
    result = normalise_windows(result)

result = np.array(result)

row = round(0.9 * result.shape[0])
train = result[:int(row), :]
np.random.shuffle(train)
X_train = train[:, :-1]
y_train = train[:, -1]
y_train = y_train[:,ticker_dict["AMZN"]] # Extract QQQ Only
X_test = result[int(row):, :-1]
y_test = result[int(row):, -1]
y_test = y_test[:,ticker_dict["AMZN"]] # Extract QQQ Only


#Step 2 Build Model
model = Sequential()

model.add(LSTM(
    input_dim=data.shape[1],
    output_dim=LAYERS,
    return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
    LAYERS,
    return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(
    output_dim=1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='adam')
print('compilation time : ', time.time() - start)

earlyStopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='min')
mcp_save = ModelCheckpoint('./forecast/models/'+MODELNAME+'_best.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, verbose=1, epsilon=1e-4, mode='min')

# Obtain Previous Val Loss Metric
model_files = os.listdir('./forecast/models/')
report_files = os.listdir('./reports/')
previous_val_loss = 1000000
previous_best_model = MODELNAME+'_best.hdf5'
if previous_best_model in model_files and FILENAME+".txt" in report_files:
    print("Previous Model Exists")
    pm = open('./reports/'+FILENAME+".txt", "r")
    pm_lines = pm.readlines()
    relevant_line = [l for l in pm_lines if "VAL_LOSS" in l]
    previous_val_loss = float(relevant_line[0].split(' ')[1])
    pm.close()
    print("Previous VAL LOSS", str(previous_val_loss))
else:
    print("No Previous Model Exists")

#Step 3 Train the model
history = model.fit(
    X_train,
    y_train,
    batch_size=512,
    nb_epoch=EPOCHS,
    validation_split=VALIDATIONSIZE,
    callbacks=[mcp_save, reduce_lr_loss, earlyStopping])

#Step 4 - Plot the predictions!
predictions = lstm.predict_sequences_multiple(model, X_test, time_horizon, time_horizon)
lstm.plot_results_multiple(predictions, y_test, time_horizon, MODELNAME+'_performance')

# Make Prediction
predictions_r = lstm.predict_sequences_multiple(model, X_test[-6:-1], time_horizon, time_horizon)
lstm.plot_results_multiple(predictions_r, predictions_r, time_horizon, MODELNAME+'_prediction')
print(predictions_r)
a = predictions_r[0][0]
b = predictions_r[0][-1]
max_a = max(predictions_r[0])
min_b = min(predictions_r[0])
pct_change = (b - a) / a * 100
max_pct_change = (max_a - min_b)/max_a * 100
slope = (b - a) / TIMEHORIZON

# Compare previous model performance to new model performance
if history.history["val_loss"][-1] < previous_val_loss:
    print("New Model Better", str(history.history["val_loss"][-1]), str(previous_val_loss))
    model.save('./forecast/models/'+MODELNAME+'_auto.h5')
    f = open('./reports/'+FILENAME+".txt", "w")
    f.write("LOSS: " + str(history.history['loss'][-1]) + "\n" + "VAL_LOSS: " + str(history.history['val_loss'][-1]) + "\n" + "PCT CHANGE: " + str(pct_change) + "\n" + "MAX PCT CHANGE: " + str(max_pct_change) + "\n" + "SLOPE: " + str(slope))
    f.close()
else:
    print("Previous Model Better", str(history.history["val_loss"][-1]), str(previous_val_loss))
    print("Making Prediction Using Previous Model")
    model = load_model('./forecast/models/'+previous_best_model)
