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
        # values
        self.d_model= d_model
        
        # prepare value embedding
        self.value_embedding= layers.Dense(d_model)
        
        # prepare time feature embedding 
        self.hour_emb= layers.Embedding(input_dim= 24, output_dim= d_model)
        self.week_emb= layers.Embedding(input_dim= 7, output_dim= d_model)
        self.month_emb= layers.Embedding(input_dim= 12, output_dim= d_model)
        
        # prepare Dropout
        self.dropout= layers.Dropout(dropout)
        
    def build(self, input_shape):
        super(Embedding, self).build(input_shape)
        
        # prepare positional embedding
        self.positional_embedding= self.add_weight(
            name="pos_emb", 
            shape=(1, 1000, self.d_model), 
            initializer='random_normal',
            trainable=True
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
                 num_heads,
                 mask= False,
                 cross= False,
                 c= 5,
                 dropout= 0.1,
                 **kwargs):
        super(ProbSparse, self).__init__(**kwargs)
        self.input_conv= layers.Conv1D(kernel_size= 3, 
                                       filters=d_model,
                                       padding='same', 
                                       activation='elu')
        self.mask= mask
        self.cross= cross
        self.c= c
        self.input_emb= Embedding(d_model=d_model)
        self.d_model= d_model
        self.num_heads= num_heads
        self.d_k= d_model // num_heads
        
        # layers
        self.dropout= layers.Dropout(dropout)
        self.dence= layers.Dense(d_model)
        
        # generate Q,K,V
        self.wq= layers.Dense(d_model)
        self.wk= layers.Dense(d_model)
        self.wv= layers.Dense(d_model)
    
    def build(self, input_shape):
        super(ProbSparse, self).build(input_shape)
    
    def call(self,
              values,
              times,
              context= None,
              context_time= None):
        
        if self.cross and context is not None:
            # generate input 
            input_emb= self.input_emb(values, times)
            input_val= self.input_conv(input_emb)
            Q= self.wq(input_val)
            
            # start ProbSparse(full attention)
            context_emb= self.input_emb(context, context_time)
            context= self.input_conv(context_emb)
            K= self.wk(context)
            V= self.wv(context)
            
        else:    
            # generate input 
            input_emb= self.input_emb(values, times)
            input_val= self.input_conv(input_emb)
            
            # start ProbSparse(full attention)
            Q= self.wq(input_val)
            K= self.wk(input_val)
            V= self.wv(input_val)
        
        # reshape & transpose
        batch_size= tf.shape(input_val)[0]
        Q= tf.reshape(Q, (batch_size, -1, self.num_heads, self.d_k))
        Q= tf.transpose(Q, perm=[0, 2, 1, 3])
        
        K= tf.reshape(K, (batch_size, -1, self.num_heads, self.d_k))
        K= tf.transpose(K, perm=[0, 2, 1, 3])
        
        V= tf.reshape(V, (batch_size, -1, self.num_heads, self.d_k))
        V= tf.transpose(V, perm=[0, 2, 1, 3])
        
        # sampling  
        query_len= tf.shape(Q)[2]
        m= tf.cast(query_len, tf.float32)
        u= tf.minimum(tf.cast(self.c * tf.math.log(m), tf.int32), query_len)
        u= tf.cast(u, tf.int32)
        
        n= tf.shape(K)[2] # number of key
        U= tf.cast(m * tf.math.log(tf.cast(n, tf.float32)), tf.int32)
        
        # select shuffle keys idx
        idx= tf.random.shuffle(tf.range(n))[:U]
        K_sample = tf.gather(K, idx, axis=2)
        
        # compute score
        scores_sample = tf.einsum("bhld,bhmd->bhlm", Q, K_sample)
        
        # sparsity
        M= tf.reduce_max(scores_sample, axis=-1) - tf.reduce_mean(scores_sample, axis=-1)

        # top u
        top_values, top_idx= tf.math.top_k(M, k=u, sorted=False)
        u= tf.shape(top_idx)[-1]
        batch_size = tf.shape(Q)[0]

        # build batch_idx and head_idx with shape (B, H, u)
        batch_idx = tf.reshape(tf.range(batch_size, dtype=tf.int32), (batch_size, 1, 1))       # (B,1,1)
        batch_idx = tf.broadcast_to(batch_idx, [batch_size, self.num_heads, u])               # (B,H,u)

        head_idx = tf.reshape(tf.range(self.num_heads, dtype=tf.int32), (1, self.num_heads, 1))  # (1,H,1)
        head_idx = tf.broadcast_to(head_idx, [batch_size, self.num_heads, u])                    # (B,H,u)

        # ensure top_idx is int32
        top_idx = tf.cast(top_idx, tf.int32) 
        gather_idx= tf.stack([batch_idx, head_idx, top_idx], axis=-1)
        
        # attention mechnism
        top_Q= tf.gather_nd(Q, gather_idx)
        scores= tf.einsum("bhud, bhnd -> bhun", top_Q, K) / tf.math.sqrt(tf.cast(self.d_k, tf.float32))
        # masking
        if self.mask:
            seq_len_q= tf.shape(scores)[-2]
            seq_len_k= tf.shape(scores)[-1]
            mask= tf.linalg.band_part(tf.ones((seq_len_q, seq_len_k)), -1, 0)
            mask= 1.0 - mask
            scores -= 1e9 * mask
        attn= tf.nn.softmax(scores, axis=-1)
        output_top= tf.einsum("bhun, bhnd -> bhud", attn, V)
        
        # get mean for another V
        V_mean= tf.reduce_mean(V, axis=2, keepdims=True)
        seq_len= tf.shape(Q)[2]
        output = tf.tile(V_mean, [1, 1, seq_len, 1])

        output = tf.tensor_scatter_nd_update(
                    output,
                    tf.reshape(gather_idx, [-1, 3]),
                    tf.reshape(output_top, [-1, self.d_k])
                )
        
        output = tf.transpose(output, perm=[0, 2, 1, 3])  
        output = tf.reshape(output, (batch_size, seq_len, self.d_model))  
        return self.dropout(output)
        
        
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
    

    probsparse_layer = ProbSparse(d_model=d_model, num_heads=2)
    output = probsparse_layer(values, times)