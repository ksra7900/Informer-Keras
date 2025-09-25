import tensorflow  as tf
from keras import layers
from keras import activations

class Distilling(layers.Layer):
    def __init__(self, d_model, **kwargs):
        super(Distilling, self).__init__(**kwargs)
        self.conv_layer= layers.Conv1D(kernel_size= 3, 
                                       filters=d_model,
                                       padding='same', 
                                       activation='elu')
        
        self.pooling_layer= layers.MaxPool1D(pool_size=2,   
                                             strides= 2)
    
    def call(self, inputs):
        x= self.conv_layer(inputs)
        x= self.pooling_layer(x)
        
        return x
    
class Embedding(layers.Layer):
    def __init__(self,
                 d_model= 512,
                 dropout= 0.1,
                 **kwargs):
        super(Embedding, self).__init__(**kwargs)
        
        # prepare value embedding
        self.value_embedding= layers.Dense(d_model)
        
        # prepare time feature embedding 
        self.hour_emb= layers.Embedding(input_dim= 24, output_dim= d_model)
        self.week_emb= layers.Embedding(input_dim= 7, output_dim= d_model)
        self.month_emb= layers.Embedding(input_dim= 12, output_dim= d_model)
        self.dropout= layers.Dropout(dropout)
        
    def build(self, input_shape):
        super(Embedding, self).build(input_shape)
        
        # prepare positional embedding
        self.positional_embedding= self.add_weight(
            "pos_emb", 
            shape=(1, 1000, d_model), 
            initializer='random_normal'
            )
        
    def call(self, 
             values,
             times):
        # value embedding
        value_emb= self.value_embedding(values)
        
        # prepare data for sequence embedding
        seq_len= tf.shape(values)[1]
        
        # time feature embedding 
        hour= self.hour_emb(times[:,:,0])
        weekday= self.week_emb(times[:,:,1])
        month= self.month_emb(times[:,:,2])
        time_emb= hour + weekday + month
        
        # combine embeddings
        combined_emb= value_emb + self.positional_embedding[:, :seq_len, :] + time_emb
        
        return self.dropout(combined_emb)
    
class ProbSparse(layers.Layer):
    def __init__(self, 
                 d_model,
                 **kwargs):
        super(ProbSparse, self).__init__(**kwargs)
        self.input_conv= layers.Conv1D(kernel_size= 3, 
                                       filters=d_model,
                                       padding='same', 
                                       activation='elu')
        self.input_emb= Embedding(d_model=d_model)
    
    def build(self, input_shape):
        super(ProbSparse, self).build(input_shape)
    
    def call(self,
              values,
              times,):
        # generate input 
        input_emb= self.input_emb(values, times)
        input_val= self.input_conv(input_emb)
        
        
        
    
    
if __name__ == '__main__':
    batch_size, L, d_model = 2, 8, 4
    
    # ورودی‌های مجزا برای values و times
    values = tf.random.normal((batch_size, L, 1))  # مقادیر float
    times = tf.cast(tf.random.uniform(
        (batch_size, L, 3), 
        minval=0, 
        maxval=[24, 7, 12]
    ), tf.int32)
    
    print("Values shape:", values.shape)
    print("Times shape:", times.shape)
    
    # تست ProbSparse با ورودی‌های مجزا
    probsparse_layer = ProbSparse(d_model=d_model)
    
    # فراخوانی با دو ورودی جداگانه
    output = probsparse_layer(values, times)