import tensorflow as tf
import keras
from keras import layers
from encoder import Encoder
from decoder import Decoder

class Informer(layers.Layer):
    def __init__(self,
                 d_model,
                 num_heads,
                 d_ff= 512,
                 dropout= 0.1,
                 c= 5,
                 **kwargs):
        super(Informer, self).__init__(**kwargs)