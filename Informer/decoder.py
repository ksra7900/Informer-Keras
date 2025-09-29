import tensorflow as tf
import keras
from keras import layers
from Informer.layers import ProbSparse

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
        self.input_projection= layers.Dense(d_model)
        
    def call(self, value, enc_value, time, enc_time):
        value_proj= self.input_projection(value)
        # self attention (masked)
        attn1= self.self_attn(value_proj, time)
        out1= self.norm_layer1(value_proj + attn1)
        
        # cross attention
        attn2 = self.cross_attn(out1, time, context=enc_value, context_time=enc_time)
        out2= self.norm_layer2(out1 + attn2)
        
        # feed forward network
        ffn= self.ffn(out2)
        output= self.norm_layer3(out2 + ffn)
        
        return self.dropout(output)
    
if __name__ == "__main__":

    batch_size = 2
    L = 8          # طول ورودی encoder
    out_len = 8    # طول ورودی decoder
    d_model = 4    # تعداد feature ها (ابعاد ورودی)

    # ورودی انکودر (values, times)
    enc_values = tf.random.normal((batch_size, L, d_model))
    enc_times = tf.random.uniform((batch_size, L, 3), maxval=24, dtype=tf.int32)

    # ورودی دیکودر (values, times)
    dec_values = tf.random.normal((batch_size, out_len, d_model))
    dec_times = tf.random.uniform((batch_size, out_len, 3), maxval=24, dtype=tf.int32)

    # تست Decoder
    decoder = Decoder(d_model=d_model, num_heads=2)
    out = decoder(dec_values, enc_values, dec_times, enc_times)

    print("Input shape (dec_values):", dec_values.shape)
    print("Input shape (enc_values):", enc_values.shape)
    print("Output shape:", out.shape)


        
        