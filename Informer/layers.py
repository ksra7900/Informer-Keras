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
        
        self.polling_layer= layers.MaxPool1D(pool_size=2,   
                                             strides= 2)
    
    def call(self, inputs):
        x= self.conv_layer(inputs)
        x= self.polling_layer(x)
        
        return x
    
class embedding(layers.Layer):
    def __init__(self,
                 d_model= 512,
                 dropout= 0.1,
                 **kwargs):
        # prepare value embedding
        self.value_embedding= layers.Dense(d_model)
        # prepare positional embedding
        self.positional_embedding= self.add_weight(
            "pos_emb", shape=(1, 1000, d_model), initializer='random_normal'
            )
        # prepare time feature embedding 
        self.hour_emb= layers.Embedding(input_dim= 24, output_dim= d_model)
        self.week_emb= layers.Embedding(input_dim= 7, output_dim= d_model)
        self.month_emb= layers.Embedding(input_dim= 12, output_dim= d_model)
        self.dropout= layers.Dropout(dropout)
        
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
    
class ProbSparce(layers.Layer):
    def __init__(self, 
                 d_model,
                 **kwargs):
        self.input_conv= layers.Conv1D(kernel_size= 3, 
                                       filters=d_model,
                                       padding='same', 
                                       activation='elu')
        self.input_emb= embedding()
    
    def build(self,
              values,
              times,):
        # generate input 
        input_conv= self.input_conv(values)
        input_emb= self.input_emb(values, times)
        input_val= input_conv + input_emb 
        
    
    def call(self, inputs):
        pass
if __name__ == '__main__':
    batch_size, L, d_model = 2, 8, 4   # مقادیر کوچیک برای تست
    x = tf.random.normal((batch_size, L, d_model))
    
    distill_layer = Distilling(d_model=d_model)
    output = distill_layer(x)
    
    print("input: ", x.shape)
    print("output: ", output.shape)
    
    # چند نمونه عددی برای بررسی
    print("input sample: \n")
    print(x[0, :5])   # 5 تای اول از نمونه اول
    print("output sample: \n")
    print(output[0, :5])   # 5 تای اول از خروجی نمونه اول
    