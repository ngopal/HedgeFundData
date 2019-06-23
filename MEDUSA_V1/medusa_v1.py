import keras
import pandas as pd
import numpy as np
from keras.layers import *
from keras.activations import *
from keras.callbacks import *
from keras.optimizers import *
from keras.models import *
import matplotlib
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from scipy import signal
from urllib.request import *
from io import StringIO
import pandas as pd
import datetime
import sys

days = 5
days_lookback = -1
LAYERS = 30
VALIDATIONSIZE = 0.10
EPOCHS = 5000
current_date_str = str(datetime.datetime.now().isoformat().split('T')[0])
MODELNAME = 'multiplemodeltest_medusa_itemized_'+current_date_str+'v1.0.1a'
matplotlib.use('Agg')


# Function to display the target and prediciton
def testmodel(epoch, logs):
    days = 5
    t = "QQQ"
    plt.style.use('fivethirtyeight')
    plt.title(t+" "+str(epoch))
    plt.plot([i[ticker_lookup[t]] for i in model.predict(np.reshape(data[-days:], (days, 1, data.shape[1])))])
    plt.xticks(np.arange(5), ('M', 'T', 'W', 'Th', 'F'))
    plt.show()

    plt.title("Volatility "+t+" "+str(epoch))
    plt.plot([i[ticker_lookup["volatility"+t]] for i in model.predict(np.reshape(data[-days:], (days, 1, data.shape[1])))])
    plt.xticks(np.arange(5), ('M', 'T', 'W', 'Th', 'F'))
    plt.show()

def df_from_fred(setname):
    # Make GET Request
    response = urlopen(url_for(setname))
    # Read response data
    data = response.read()
    # Convert binary text to utf-8
    text = data.decode('utf-8')
    # Convert text file to pandas dataframe
    TEXTDATA = StringIO(text)
    df = pd.read_csv(TEXTDATA, sep=",")
    df = df.set_index("DATE")
    return df

def url_for(series):
    """function takes FRED series name as input. For example, GDPC1, or HOUST."""
    print(series)
    return "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=968&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id="+series+"&scale=left&cosd=1947-01-01&coed=2019-01-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Quarterly&fam=avg&fgst=lin&fgsnd=2009-06-01&line_index=1&transformation=lin&vintage_date=2019-05-07&revision_date=2019-05-07&nd=1947-01-01"

data_orig = pd.read_csv('./data/files/multiple_concatenated_tickers.csv')
vol_data_orig = pd.read_csv('./data/files/multiple_concatenated_tickers_volatility.csv')

data_orig = data_orig\
  .merge(vol_data_orig, how="inner", left_on=data_orig.Index, right_on=vol_data_orig.Index).fillna(method="ffill")\
  .drop(["key_0", "Index_y"], axis=1)\
  .rename(columns={'Index_x':'Index'})

# Vital Economic indicators: https://gist.github.com/ngopal/262fce10a7fa4a8467f0c61a13c85dc5
GDPC1 = df_from_fred("GDPC1")
time.sleep(5)
M2 = df_from_fred("M2")
time.sleep(5)
CPALTT01USQ657N = df_from_fred("CPALTT01USQ657N")
time.sleep(5)
PPIACO = df_from_fred("PPIACO")
time.sleep(5)
UMCSENT = df_from_fred("UMCSENT")
time.sleep(5)
PAYEMS = df_from_fred("PAYEMS")
time.sleep(5)
RRSFS = df_from_fred("RRSFS")
time.sleep(5)
HOUST = df_from_fred("HOUST")
time.sleep(5)
ISRATIO = df_from_fred("ISRATIO")
time.sleep(5)
SP500 = df_from_fred("SP500")
time.sleep(5)

# Thoughts
FEDFUNDS = df_from_fred("FEDFUNDS")
time.sleep(5)
UNRATE = df_from_fred("UNRATE")
time.sleep(5)
T10Y2Y = df_from_fred("T10Y2Y")
time.sleep(5)
CBBTCUSD = df_from_fred("CBBTCUSD")
time.sleep(5)

# Others
IPMAN = df_from_fred("IPMAN")
time.sleep(5)
MPU9900063 = df_from_fred("MPU9900063")
time.sleep(5)
PCU33443344 = df_from_fred("PCU33443344")
time.sleep(5)
MEHOINUSA672N = df_from_fred("MEHOINUSA672N")
time.sleep(5)
TCMDO = df_from_fred("TCMDO")
time.sleep(5)
FGTCMDODNS = df_from_fred("FGTCMDODNS")
time.sleep(5)
ADSLFAA027N = df_from_fred("ADSLFAA027N")
time.sleep(5)
NCBCMDPMVCE = df_from_fred("NCBCMDPMVCE")
time.sleep(5)
FGCCSAQ027S = df_from_fred("FGCCSAQ027S")
time.sleep(5)
ASTNITA = df_from_fred("ASTNITA")
time.sleep(5)
PCETRIM12M159SFRBDAL = df_from_fred("PCETRIM12M159SFRBDAL")
DAAA = df_from_fred("DAAA")
time.sleep(5)
USSLIND = df_from_fred("USSLIND")
time.sleep(5)
IRLTLT01USM156N = df_from_fred("IRLTLT01USM156N")
time.sleep(5)

CPGDPAI = df_from_fred("CPGDPAI")
time.sleep(5)
DCOILWTICO = df_from_fred("DCOILWTICO")
time.sleep(5)
PNRGINDEXM = df_from_fred("PNRGINDEXM")
time.sleep(5)
PCU3121123121120 = df_from_fred("PCU3121123121120")
time.sleep(5)

data_orig2 = data_orig\
  .join(GDPC1)\
  .join(M2)\
  .join(CPALTT01USQ657N)\
  .join(PPIACO)\
  .join(UMCSENT)\
  .join(PAYEMS)\
  .join(RRSFS)\
  .join(HOUST)\
  .join(ISRATIO)\
  .join(SP500)\
  .join(FEDFUNDS)\
  .join(UNRATE)\
  .join(T10Y2Y)\
  .join(CBBTCUSD)\
  .join(IPMAN)\
  .join(MPU9900063)\
  .join(PCU33443344)\
  .join(MEHOINUSA672N)\
  .join(TCMDO)\
  .join(FGTCMDODNS)\
  .join(ADSLFAA027N)\
  .join(NCBCMDPMVCE)\
  .join(FGCCSAQ027S)\
  .join(ASTNITA)\
  .join(PCETRIM12M159SFRBDAL)\
  .join(DAAA)\
  .join(USSLIND)\
  .join(IRLTLT01USM156N)\
  .join(CPGDPAI)\
  .join(DCOILWTICO)\
  .join(PNRGINDEXM)\
  .join(PCU3121123121120)\
  .fillna(method="ffill")\
  .fillna(-1)

ticker_lookup = dict([(i[1].split('.')[0], int(i[0])) for i in enumerate(list(data_orig.columns)) if (('Open' in i[1]) or ("volatility" in i[1]))])
inv_map = {v+'.Open': k for k, v in dict(enumerate(list(ticker_lookup.keys()))).items()}

pct_df = data_orig.set_index("Index").shift(days_lookback).dropna()

# De-trend data
data_orig = data_orig.set_index("Index")
data_orig_detrended = signal.detrend(data_orig)

# Normalized data
scaler = MinMaxScaler(feature_range=(0, 1))
data_mat = scaler.fit_transform(data_orig_detrended[0:(data_orig_detrended.shape[0] - 1)])
pctdf_mat = scaler.transform(pct_df)

seq_len = 90 # days to use for prediction
data = np.array((data_mat))

sequence_length = seq_len + 1
result = []
for index in range(len(data) - sequence_length):
    result.append(data[index: index + sequence_length])

result = np.array(result)
row = round(0.9 * result.shape[0])

pctdf_mat = pctdf_mat[sequence_length:]

x_train = result[:int(row), :]
y_train = pctdf_mat[:int(row),:] 

x_test = result[int(row):, :]
y_test = pctdf_mat[int(row):,:]  

model = Sequential()

model.add(LSTM(
    input_dim=data.shape[1],
    output_dim=LAYERS,
    kernel_regularizer=keras.regularizers.l2(0.01),
    return_sequences=True))
model.add(BatchNormalization())
model.add(LSTM(
    LAYERS,
    kernel_regularizer=keras.regularizers.l2(0.01),
    return_sequences=False))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(
    output_dim=y_train.shape[1]))
model.add(Activation('linear'))

start = time.time()
adam = Adam(lr=0.1)
model.compile(loss='mse', optimizer=adam)
print('compilation time : ', time.time() - start)


mcp_save = ModelCheckpoint('./models/'+MODELNAME+'_best.hdf5', save_best_only=True, monitor='val_loss', mode='min')
tbrd = TensorBoard(log_dir='./models/logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

# No need for this in production mode
# testmodelcb = keras.callbacks.LambdaCallback(on_epoch_end=testmodel)

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, verbose=1, epsilon=1e-4, mode='min')
reduce_lr_loss_training = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=15, verbose=1, epsilon=1e-4, mode='min')

args = sys.argv[1:]
if args[0] == "--run-only":
    best_model = keras.models.load_model('./models/'+MODELNAME+'_best.hdf5') 

    for t in list(ticker_lookup.keys()):
        plt.style.use('fivethirtyeight')
        plt.title(str(t)+" for week of "+current_date_str)
        plt.plot([i[ticker_lookup[t]] for i in best_model.predict(np.reshape(data[-days:], (days, 1, data.shape[1])))])
        plt.xticks(np.arange(5), ('M', 'T', 'W', 'Th', 'F'))
        if "volatility" in t:
            plt.savefig('./reports/'+t.split("volatility")[1]+'_volatility_prediction.png')
        else:
            plt.savefig('./reports/'+t+'_prediction.png')
        plt.show()
else:
    history = model.fit(
        x_train,
        y_train,
        batch_size=1024,
        nb_epoch=EPOCHS,
        validation_split=VALIDATIONSIZE,
        callbacks = [reduce_lr_loss, mcp_save, tbrd, reduce_lr_loss_training],
        shuffle=True)

    best_model = keras.models.load_model('./models/'+MODELNAME+'_best.hdf5') 

    for t in list(ticker_lookup.keys()):
        plt.style.use('fivethirtyeight')
        plt.title(str(t)+" for week of "+current_date_str)
        plt.plot([i[ticker_lookup[t]] for i in best_model.predict(np.reshape(data[-days:], (days, 1, data.shape[1])))])
        plt.xticks(np.arange(5), ('M', 'T', 'W', 'Th', 'F'))
        if "volatility" in t:
            plt.savefig('./reports/'+t.split("volatility")[1]+'_volatility_prediction.png')
        else:
            plt.savefig('./reports/'+t+'_prediction.png')
        plt.show()