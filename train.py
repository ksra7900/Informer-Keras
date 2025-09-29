import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from keras import models
from keras import layers
import tensorflow as tf
from Informer import Informer
import warnings
warnings.filterwarnings('ignore')

enc_len = 144   # طول ورودی انکودر (یک روز گذشته)
dec_len = 24    # طول ورودی دیکودر (یک روز آینده)

def create_enc_dec_sequences(X, y, date, enc_len, dec_len):
    enc_X, enc_time, dec_X, dec_time, target = [], [], [], [], []
    for i in range(len(X) - enc_len - dec_len):
        enc_X.append(X[i:i+enc_len])
        enc_time.append(date[i:i+enc_len])
        dec_X.append(X[i+enc_len:i+enc_len+dec_len])   # اینجا میشه تارگت‌های شیفت‌شده
        dec_time.append(date[i+enc_len:i+enc_len+dec_len])
        target.append(y[i+enc_len:i+enc_len+dec_len])  # پیش‌بینی dec_len گام جلو
    return (np.array(enc_X), np.array(enc_time),
            np.array(dec_X), np.array(dec_time),
            np.array(target))

# read dataset
df= pd.read_csv('Data/Tetuan City power consumption.csv')

# create date dataset
df_date= pd.DataFrame()
df['DateTime']= pd.to_datetime(df['DateTime'])
df_date['minute']= df['DateTime'].dt.minute
df_date['hour']= df['DateTime'].dt.hour
df_date['weekday']= df['DateTime'].dt.weekday
df_date['month']= df['DateTime'].dt.month

# create and prepare data & target 
df_data= df.drop(['DateTime', 'Zone 1 Power Consumption'], axis=1)
df_target= df[['Zone 1 Power Consumption']]

# power scaler
Power_scaler= PowerTransformer(method='box-cox')
df_data['normed_general diffuse flows'] = Power_scaler.fit_transform(df_data[['general diffuse flows']])
df_data['normed_diffuse flows'] = Power_scaler.fit_transform(df_data[['diffuse flows']])
df_data['normed_humidity'] = np.clip(df_data['Humidity'], a_min=40, a_max=90)
df_data = df_data.drop(['Humidity', 'general diffuse flows', 'diffuse flows'], axis=1)

# MinMax scaler
MinMax_scaler= MinMaxScaler(feature_range=(0, 1))
df_data= MinMax_scaler.fit_transform(df_data)
df_target= MinMax_scaler.fit_transform(df_target)

# sequencing data
enc_X, enc_time, dec_X, dec_time, target = create_enc_dec_sequences(
    df_data, df_target, df_date.values, enc_len, dec_len
)

enc_time_df = pd.DataFrame(enc_time.reshape(-1, enc_time.shape[-1]),
                           columns=['minute','hour','weekday','month'])
enc_time_used = enc_time_df[['hour','weekday','month']].values.reshape(enc_time.shape[0], enc_time.shape[1], 3)

dec_time_df = pd.DataFrame(dec_time.reshape(-1, dec_time.shape[-1]),
                           columns=['minute','hour','weekday','month'])
dec_time_used = dec_time_df[['hour','weekday','month']].values.reshape(dec_time.shape[0], dec_time.shape[1], 3)


dataset = tf.data.Dataset.from_tensor_slices(((enc_X, enc_time_used, dec_X, dec_time_used), target))

# train/val/test split
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))

train_ds = dataset.take(train_size).batch(32).shuffle(100)
val_ds = dataset.skip(train_size).take(val_size).batch(32)
test_ds = dataset.skip(train_size + val_size).batch(32)

# تعریف مدل
model = Informer(d_model=64, num_heads=4)  # مقادیر رو بسته به منابع تغییر بده
model.compile(optimizer='adam', loss='mse')

# آموزش
history = model.fit(train_ds, validation_data=val_ds, epochs=10, batch_size=32)

# ارزیابی
model.evaluate(test_ds)