import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from keras import models
from keras import layers
import tensorflow as tf
from Informer import Informer

# sequencing
def create_sequences(X, y, window_size):
    seq_X = []
    seq_y = []
    for i in range(len(X) - window_size):
        seq_X.append(X[i:i+window_size])
        seq_y.append(y[i+window_size])

    return np.array(seq_X), np.array(seq_y)

# read dataset
df= pd.read_csv('Data/Tetuan City power consumption.csv')

# create date dataset
df_date= df[['DateTime']]
df_date['DateTime']= pd.to_datetime(df['DateTime'])
df_date['minute']= df_date['DateTime'].dt.minute
df_date['hour']= df_date['DateTime'].dt.hour
df_date['weekday']= df_date['DateTime'].dt.weekday
df_date['month']= df_date['DateTime'].dt.month


# create & prepare data&target dataset
df_data= df.drop(['DateTime', 'Zone 1 Power Consumption'], axis=1)
target= df[['Zone 1 Power Consumption']]

# power scaler
Power_scaler= PowerTransformer(method='box-cox')
df_data['normed_general diffuse flows'] = Power_scaler.fit_transform(df_data[['general diffuse flows']])
df_data['normed_diffuse flows'] = Power_scaler.fit_transform(df_data[['diffuse flows']])
df_data['normed_humidity'] = np.clip(df_data['Humidity'], a_min=40, a_max=90)
df_data = df_data.drop(['Humidity', 'general diffuse flows', 'diffuse flows'], axis=1)

# MinMax scaler
MinMax_scaler= MinMaxScaler(feature_range=(0, 1))
df_data= MinMax_scaler.fit_transform(df_data)
target= MinMax_scaler.fit_transform(target)

# sequencing data
data, target= create_sequences(df_data, target, window_size=144)