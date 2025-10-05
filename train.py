import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
import keras
from keras import optimizers
from keras.callbacks import EarlyStopping
import tensorflow as tf
from Informer import Informer
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

keras.backend.clear_session()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# 144
enc_len = 1008   # lenght of encoder (past days)
dec_len = 24    # lenght of decoder (future day)

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

train_ds = dataset.take(train_size).batch(2)
val_ds = dataset.skip(train_size).take(val_size).batch(2)
test_ds = dataset.skip(train_size + val_size).batch(2)

# early stopping
callback= EarlyStopping(monitor='val_loss',
                        patience=20,
                        restore_best_weights= True)

# define model
model = Informer(d_model=32, 
                 num_heads=2,
                 e_layers=1,
                 d_layers=1
                 )
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='MAE',
              metrics=['accuracy'])

# learn
history = model.fit(train_ds, 
                    validation_data=val_ds, 
                    epochs=1000, 
                    callbacks=[callback]
                    )

# evaluation
model.evaluate(test_ds)
#model.save('model.h5')

# prediction & visualization
y_pred= model.predict(test_ds)
y_true = np.concatenate([y for x, y in test_ds], axis=0)

y_pred_rescaled = MinMax_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
y_true_rescaled = MinMax_scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(y_true.shape)

plt.figure(figsize=(12,6))
plt.plot(y_true_rescaled[0], label="Actual", marker='o')
plt.plot(y_pred_rescaled[0], label="Predicted", marker='x')
plt.title("Test Sequence Prediction vs Actual")
plt.xlabel("Time step")
plt.ylabel("Target value")
plt.legend()
plt.show()

# visualize loss & val_loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10,6))
plt.plot(train_loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()