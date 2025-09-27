import tensorflow as tf
import keras
from keras import layers
from layers import ProbSparse, Distilling

class Encoder(layers.Layer):
    def __init__(self,
                 d_model,
                 num_heads,
                 d_ff= 512,
                 dropout= 0.1,
                 c= 5,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)
        
        # values
        self.c= c
        self.num_heads= num_heads
        self.d_model= d_model
        
        # layers
        self.attn_layer= ProbSparse(d_model= d_model, 
                                    num_heads= num_heads,
                                    c=c,
                                    dropout=dropout)
        self.distilling= Distilling(d_model= d_model)
        self.ffn= keras.models.Sequential(
            [
                layers.Dense(d_ff, activation='relu'),
                layers.Dense(d_model)
                ]
            )
        self.norm_layer1= layers.LayerNormalization(epsilon=1e-6)
        self.norm_layer2= layers.LayerNormalization(epsilon=1e-6)
        self.dropout= layers.Dropout(dropout)
        
    def call(self, value, time):
        # ProbSparse Attention
        attn_layer= self.attn_layer(value, time) 
        out1= self.norm_layer1(value + attn_layer)
        
        # FeedForward
        ffn_output= self.ffn(out1)
        output_attn= self.norm_layer2(out1 + ffn_output)
        
        # Disttiling layer
        distilling= self.distilling(output_attn)
        
        # output
        output= self.dropout(distilling)
        
        return output
    
if __name__ == '__main__':
    batch_size, L, d_model = 2, 8, 4
    
    # inputs
    values = tf.random.normal((batch_size, L, 1))  # مقادیر float
    times = tf.cast(tf.random.uniform(
        (batch_size, L, 3), 
        minval=0, 
        maxval=[24, 7, 12]
    ), tf.int32)
    
    print("Values shape:", values.shape)
    print("Times shape:", times.shape)
    
    # inputs
    values_in = layers.Input(shape=(L, 1))
    times_in = layers.Input(shape=(L, 3))
    
    # model
    encoder_layer = Encoder(d_model=d_model, num_heads=2)
    x = encoder_layer(values_in, times_in)
    output = layers.Dense(1)(x)
    
    # prepare model
    model = keras.models.Model(inputs=[values_in, times_in], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    