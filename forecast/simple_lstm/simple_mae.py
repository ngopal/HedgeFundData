import sys, os
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import lstm, time #helper libraries

FILENAME = sys.argv[1]
MODELNAME = FILENAME.split('.')[0]

time_horizon = 5
X_train, y_train, X_test, y_test = lstm.load_data('./data/files/'+FILENAME+'_open.csv', time_horizon, True)

#Step 2 Build Model
model = Sequential()

model.add(LSTM(
    input_dim=1,
    output_dim=100,
    return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
    100,
    return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(
    output_dim=1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss='mae', optimizer='adam')
print('compilation time : ', time.time() - start)

# earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('./forecast/models/'+MODELNAME+'_best.hdf5', save_best_only=True, monitor='val_loss', mode='min')
# reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=10, verbose=1, epsilon=1e-4, mode='min')


# Obtain Previous Val Loss Metric
model_files = os.listdir('./forecast/models/')
previous_val_loss = None
if MODELNAME+'_auto.h5' in model_files:
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
    nb_epoch=200,
    validation_split=0.20,
    callbacks=[mcp_save])

#Step 4 - Plot the predictions!
predictions = lstm.predict_sequences_multiple(model, X_test, time_horizon, time_horizon)
lstm.plot_results_multiple(predictions, y_test, time_horizon, MODELNAME+'_performance_MAE')

# Compare previous model performance to new model performance
if history.history["val_loss"][-1] < previous_val_loss:
    print("New Model Better", str(history.history["val_loss"][-1]), str(previous_val_loss))
    model.save('./forecast/models/'+MODELNAME+'_auto.h5')
else:
    print("Previous Model Better", str(history.history["val_loss"][-1]), str(previous_val_loss))
    print("Making Prediction Using Previous Model")
    model = load_model('./forecast/models/'+MODELNAME+'_auto.h5')

# Save Model
# model.save('./forecast/models/'+MODELNAME+'_auto_MAE.h5')

# Make Prediction
predictions_r = lstm.predict_sequences_multiple(model, X_test[-6:-1], time_horizon, time_horizon)
lstm.plot_results_multiple(predictions_r, predictions_r, time_horizon, MODELNAME+'_prediction_MAE')
print(predictions_r)
a = predictions_r[0][0]
b = predictions_r[0][-1]

pct_change = (b - a) / a * 100

f = open('./reports/'+FILENAME+"_MAE.txt", "w")
f.write("LOSS: " + str(history.history['loss'][-1]) + "\n" + "VAL_LOSS: " + str(history.history['val_loss'][-1]) + "\n" + "PCT CHANGE: " + str(pct_change))
f.close()
