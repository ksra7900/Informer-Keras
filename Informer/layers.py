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
    
class ProbSparce(layers.Layer):
    def __init__(self, 
                 **kwargs):
        pass
    
    def build(self, input_shape):
        pass
    
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
    