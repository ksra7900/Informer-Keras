import tensorflow as tf
import keras
from keras import layers
from encoder import Encoder
from decoder import Decoder

class Informer(layers.Layer):
    def __init__(self,
                 d_model,
                 num_heads,
                 num_encoder= 2,
                 num_decoder= 1,
                 d_ff= 512,
                 dropout= 0.1,
                 c= 5,
                 **kwargs):
        super(Informer, self).__init__(**kwargs)
        
        self.encoder_layer= [Encoder(d_model= d_model, 
                                    num_heads= num_heads,
                                    dropout= dropout,
                                    c= c) for _ in range(num_encoder)]
        
        self.decoder_layer= [Decoder(d_model= d_model, 
                                    num_heads= num_heads,
                                    dropout= dropout,
                                    c= c) for _ in range(num_decoder)]
        
        self.output_layer= layers.Dense(1)
        
    def call(self, enc_value, enc_time, dec_value, dec_time):
        # prepare value for encoder
        x= enc_value
        times= enc_time
        
        # encoder layer
        for enc in self.encoder_layer:
            x= enc(x, times)
            times= times[:, ::2, :]
        
        # encoder output
        output_enc= x
        
        # prepare value for decoder
        y= dec_value
        dec_t= dec_time
        
        # decoder layer
        for dec in self.decoder_layer:
            y= dec(y, output_enc, dec_t, enc_time)
            
        # output layer
        output= self.output_layer(y)
        return output
        

if __name__ == '__main__':
    # quick functional-API model build & test (use even lengths compatible with downsampling)
    batch_size = 2
    e_layers = 2    # number of encoder layers (each halves length)
    d_layers = 1
    L_enc = 16      # must be divisible by 2**e_layers (here 4) -> 16 OK
    L_dec = 4
    d_model = 8

    # Keras Input tensors (KerasTensors) -> so we can make a Model and summary
    enc_values_in = layers.Input(shape=(L_enc, 1), name='enc_values')
    enc_times_in = layers.Input(shape=(L_enc, 3), dtype='int32', name='enc_times')
    dec_values_in = layers.Input(shape=(L_dec, 1), name='dec_values')
    dec_times_in = layers.Input(shape=(L_dec, 3), dtype='int32', name='dec_times')

    model_core = Informer(d_model=d_model, num_heads=2, num_encoder=e_layers, num_decoder=d_layers)
    outputs = model_core(enc_values_in, enc_times_in, dec_values_in, dec_times_in)

    model = keras.Model(inputs=[enc_values_in, enc_times_in, dec_values_in, dec_times_in], outputs=outputs)
    model.compile(optimizer='adam', loss='mse')

    model.summary()

    # quick forward pass with random data to ensure runtime works
    import numpy as np
    enc_values = np.random.randn(batch_size, L_enc, 1).astype(np.float32)
    enc_times = np.random.randint(low=0, high=[24,7,12], size=(batch_size, L_enc, 3)).astype(np.int32)
    dec_values = np.random.randn(batch_size, L_dec, 1).astype(np.float32)
    dec_times = np.random.randint(low=0, high=[24,7,12], size=(batch_size, L_dec, 3)).astype(np.int32)

    out = model.predict([enc_values, enc_times, dec_values, dec_times])
    print("Output shape:", out.shape)
        
        