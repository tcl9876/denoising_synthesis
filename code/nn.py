
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Conv2D, AveragePooling2D, Dropout
from tensorflow_addons.layers import GroupNormalization
import numpy as np


def get_timestep_embedding(timesteps, embedding_dim: int):
    #timestep embedding for self attentional layers.

    assert len(timesteps.shape) == 1 

    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
    
    emb = tf.cast(timesteps, dtype=tf.float32)[:, None] * emb[None, :]
    emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = tf.pad(emb, [[0, 0], [0, 1]])
    assert emb.shape == [timesteps.shape[0], embedding_dim]
    return emb

class downsample(Layer):
    #reduces the spatial dimension by 2.
    def __init__(self, c, with_conv, name=None):
        super().__init__(name=name)
        if with_conv:
            self.down = Conv2D(c, 3, padding='same', strides=2)
        else:
            self.down = AveragePooling2D()
    
    def call(self, x, index):
        return self.down(x)
      
class upsample(Layer):
    #increases the spatial dimension by 2
    def __init__(self, c, with_conv, name=None):
        super().__init__(name=name)
        self.with_conv = with_conv
        if self.with_conv:
            self.up = Conv2D(c, 3, padding='same')

    def call(self, x, index):
        B, H, W, C = x.shape
        x = tf.image.resize(x, size=[H * 2, W * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        if self.with_conv:
            x = self.up(x)
        return x

class resnet_block(Layer):
    #residual convolutional block
    def __init__(self, c, name=None, drop_rate=0.0, groups=32):
        super().__init__(name=name) 
        self.c = c
        self.drop_rate = drop_rate
        
        self.conv1 = Conv2D(c, 3, padding='same')
        self.conv2 = Conv2D(c, 3, padding='same')
        
        self.norm1 = GroupNormalization(groups=groups) 
        self.norm2 = GroupNormalization(groups=groups)
        self.temb_proj = Dense(c) 

        if drop_rate > 0.01:
            self.drop_fn = Dropout(drop_rate)
        else:
            self.drop_fn = tf.identity

    def build(self, input_shape):
        if input_shape[-1] != self.c:
            self.skip_conv = Dense(self.c)
        else:
            self.skip_conv = None

    def call(self, x, index):
        residual = tf.identity(x)
        x = tf.nn.swish(self.norm1(x))
        x = self.conv1(x)
        x = self.drop_fn(x)
        
        x += self.temb_proj(tf.nn.swish(index))[:, None, None, :]
        x = tf.nn.swish(self.norm2(x))
        x = self.conv2(x)
        x = self.drop_fn(x)

        if self.skip_conv is not None:
            residual = self.skip_conv(residual)
        
        return x + residual      

class attn_block(Layer):
    #self attentional block
    def __init__(self, c, name=None, drop_rate=0.0, groups=32):
        super().__init__(name=name) 
        self.c = c
        self.k = Dense(c)
        self.norm = GroupNormalization(groups=groups)
        self.proj_out = Dense(c)
        self.q = Dense(c)
        self.v = Dense(c)
        
        if drop_rate > 0.01:
            self.drop_fn = Dropout(drop_rate)
        else:
            self.drop_fn = tf.identity
        
    def build(self, input_shape):
        if input_shape[-1] != self.c:
            self.skip = Dense(self.c)
        else:
            self.skip = None

    def call(self, x, index):
        B, H, W, C = x.shape
        residual = tf.identity(x)
        x = self.norm(x)
        q, k, v = self.q(x), self.k(x), self.v(x)

        w = tf.einsum('bhwc,bHWc->bhwHW', q, k) * (int(C) ** (-0.5))
        w = tf.reshape(w, [B, H, W, H * W])
        w = tf.nn.softmax(w, -1)
        w = tf.reshape(w, [B, H, W, H, W])
        x = tf.einsum('bhwHW,bHWc->bhwc', w, v)

        x = self.drop_fn(x)

        if self.skip is not None:
            residual = self.skip(residual)

        x = self.proj_out(x)
        return x + residual
