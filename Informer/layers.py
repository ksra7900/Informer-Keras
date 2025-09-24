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
        self.value_embedding= layers.Dense(d_model)
        self.positional_embedding= self.add_weight(
            "pos_emb", shape=(1, 1000, d_model), initializer='random_normal'
            )
        self.dropout= layers.Dropout(dropout)
        
    def call(self, inputs):
        value_emb= self.value_embedding(inputs)
        seq_len= tf.shape(inputs)[1]
        x= value_emb + self.positional_embedding[:, :seq_len, :]
        return self.dropout(x)
    
class ProbSparce(layers.Layer):
    def __init__(self, 
                 d_model,
                 **kwargs):
        self.input_conv= layers.Conv1D(kernel_size= 3, 
                                       filters=d_model,
                                       padding='same', 
                                       activation='elu')
        self.input_emb= embedding()
    
    def build(self, inputs):
        # generate input 
        input_conv= self.input_conv(inputs)
        input_emb= self.input_emb(inputs)
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
    