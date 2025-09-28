import tensorflow as tf
import keras
from keras import layers
from layers import ProbSparse

class Decoder(layers.Layer):
    def __init__(self,
                 d_model,
                 num_heads,
                 d_ff= 512,
                 dropout= 0.1,
                 c= 5,
                 **kwargs):
        super(Decoder, self).__init__(**kwargs)
        
        # layers
        self.self_attn= ProbSparse(d_model= d_model,
                                    num_heads= num_heads,
                                    c= c,
                                    dropout=dropout,
                                    mask=True)
        
        self.cross_attn= ProbSparse(d_model= d_model,
                                    num_heads= num_heads,
                                    c= c,
                                    dropout=dropout,
                                    cross=True)
        
        self.ffn= keras.models.Sequential(
            [
                layers.Dense(d_ff, activation='relu'),
                layers.Dense(d_model)
                ]
            )
        
        self.norm_layer1= layers.LayerNormalization(epsilon=1e-6)
        self.norm_layer2= layers.LayerNormalization(epsilon=1e-6)
        self.norm_layer3= layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout= layers.Dropout(dropout)
        
    def call(self, value, enc_value, time):
        # self attention (masked)
        attn1= self.self_attn(value, time)
        out1= self.norm_layer1(value + attn1)
        
        # cross attention
        attn2= self.cross_attn(attn1, time, context=enc_value)
        out2= self.norm_layer2(out1 + attn2)
        
        # feed forward network
        ffn= self.ffn(out2)
        output= self.norm_layer3(out2 + ffn)
        
        return self.dropout(output)
    
if __name__ == '__main__':
    batch_size, L, d_model = 2, 8, 4
    out_len = 4   # طول دنباله‌ی decoder

    # ورودی‌های encoder
    enc_values = tf.random.normal((batch_size, L, 1))
    '''enc_times = tf.cast(tf.random.uniform(
        (batch_size, L, 3),
        minval=0,
        maxval=[24, 7, 12]
    ), tf.int32)'''

    # ورودی‌های decoder (مثلاً sequence کوتاه‌تر)
    dec_values = tf.random.normal((batch_size, out_len, 1))
    dec_times = tf.cast(tf.random.uniform(
        (batch_size, out_len, 3),
        minval=0,
        maxval=[24, 7, 12]
    ), tf.int32)

    # ساخت Decoder
    decoder = Decoder(d_model=d_model, num_heads=2)

    # forward pass
    output = decoder(dec_values, enc_values, dec_times)

    print("Decoder output shape:", output.shape)

        
        