import keras
from keras import layers
from Informer.encoder import Encoder
from Informer.decoder import Decoder

class Informer(keras.Model):
    def __init__(self,
                 d_model,
                 num_heads,
                 e_layers= 2,
                 d_layers= 1,
                 d_ff= 512,
                 dropout= 0.1,
                 c= 5,
                 **kwargs):
        super(Informer, self).__init__(**kwargs)
        
        self.encoder= [Encoder(d_model= d_model, 
                                    num_heads= num_heads,
                                    dropout= dropout,
                                    c= c) for _ in range(e_layers)]
        
        self.decoder= [Decoder(d_model= d_model, 
                                    num_heads= num_heads,
                                    dropout= dropout,
                                    c= c) for _ in range(d_layers)]
        
        self.output_layer= layers.Dense(1)
        self.dropout= layers.Dropout(dropout)
        
    def call(self, inputs, training= False):
        enc_value, context_time, dec_value, dec_time= inputs
        # prepare value for encoder
        x= enc_value
        times= context_time
        
        # encoder layer
        for enc in self.encoder:
            x= enc(x, times)
            times= times[:, ::2, :]
        
        # encoder output
        output_enc= x
        
        # prepare value for decoder
        y= dec_value
        dec_t= dec_time
        
        # decoder layer
        for dec in self.decoder:
            y= dec(y, output_enc, dec_t, times)
            
        # output layer
        output= self.output_layer(y)
        return output
        
        