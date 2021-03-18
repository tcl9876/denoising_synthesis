import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras import Model
from tensorflow_addons.layers import GroupNormalization
import os
from nn import get_timestep_embedding, downsample, upsample, resnet_block, attn_block

class GenerativeModel(Model):
    def __init__(self, n_steps, c, input_shape, chmuls, layers_per_resolution, sa_list=None, drop_rate=0.0, groups=32):
        super().__init__(name="Generative_Model")
        self.c = c
        self.n_steps = n_steps

        if len(input_shape) == 4:
            input_shape = input_shape[1:]

        n_res = len(chmuls) #the number of spatial resolutions
        max_downscale_factor = int(2 ** (n_res - 1))

        assert len(input_shape) == 3, "You must provide an input shape of NHWC or HWC"
        assert n_res > 0, "Invalid value for channel multipliers. It should be a list of at least length 1."
        for i in chmuls:
            if not isinstance(i, int) and i:
                raise ValueError("The 'chmuls' argument must be a list of integers greater than 0.")

        h, w = input_shape[:2]
        nl = layers_per_resolution

        self.h = h
        self.w = w
        self.inp_shape = list(input_shape)
        self.chmuls = chmuls

        channels = [int(c*chmuls[i]) for i in range(len(chmuls))]

        if sa_list is None:
            sa_list = [False for _ in range(n_res-1)] + [True]
        else:
            sa_list = [bool(i) for i in sa_list]
            assert len(sa_list) == n_res, "The list specifying which resolutions to use self attentional layers should be of the same length as the number of spatial resolutions"

        assert h%max_downscale_factor == 0, "Height of images must be divisible by {}".format(max_downscale_factor)
        assert w%max_downscale_factor == 0, "Width of images must be divisible by {}".format(max_downscale_factor)
        assert c%groups == 0, "The channel size must be divisible by the number of groups for groupnormalization. The number of groups for groupnorm is {} while your channel is of size {}".format(groups, c)
        assert nl > 0, "You must have at least one layer per resolution."

        #in and out layers
        self.conv_in = Conv2D(c, 3, padding="same")
        self.norm_out = GroupNormalization(groups=groups)
        self.conv_out = Conv2D(3, 3, dtype=tf.float32, padding="same")
        self.temb = [Dense(c*4), Dense(c*4)]

        #the first part of the U-Net
        self.down_layers = []
        for i, c_i, use_sa in zip(range(n_res), channels, sa_list):
            lyrs = []
            names_conv = ['down_convolutional_{}_{}'.format(i, j) for j in range(nl)]
            if use_sa:
                names_attn = ['down_attentional_{}_{}'.format(i, j) for j in range(nl-1)] + [None]
            else:
                names_attn = [None] * nl
                
            for name_c, name_sa in zip(names_conv, names_attn):
                lyrs.append(resnet_block(c_i, name=name_c, drop_rate=drop_rate, groups=groups))
                if name_sa is not None:
                    lyrs.append(attn_block(c_i, name=name_sa, drop_rate=drop_rate, groups=groups))

            if i < n_res - 1:
                lyrs.append(downsample(c_i, name="downsampling_{}".format(i), with_conv=True))

            for lyr in lyrs:
                self.down_layers.append(lyr)

        #the middle, or the bottom of the U-Net
        mid_layers = [resnet_block(channels[-1], name="mid_convolutional_0", drop_rate=drop_rate, groups=groups)]
        mid_layers.append(resnet_block(channels[-1], name="mid_attentional_0", drop_rate=drop_rate, groups=groups))
        mid_layers.append(resnet_block(channels[-1], name="mid_convolutional_2", drop_rate=drop_rate, groups=groups))
        self.mid_layers = mid_layers

        self.up_layers = []
        for i, c_i, use_sa in zip(reversed(range(n_res)), reversed(channels), reversed(sa_list)):
            lyrs = []
            names_conv = ['up_convolutional_{}_{}'.format(i, j) for j in range(nl)]
            if use_sa:
                names_attn = ['up_attentional_{}_{}'.format(i, j) for j in range(nl-1)] + [None]
            else:
                names_attn = [None] * nl
                
            for name_c, name_sa in zip(names_conv, names_attn):
                lyrs.append(resnet_block(c_i, name=name_c, drop_rate=drop_rate, groups=groups))
                if name_sa is not None:
                    lyrs.append(attn_block(c_i, name=name_sa, drop_rate=drop_rate, groups=groups))

            if i > 0:
                lyrs.append(upsample(c_i, name="upsampling_{}".format(i), with_conv=True))

            for lyr in lyrs:
                self.up_layers.append(lyr)

        self.make_weights()

    def make_weights(self): #equivalent of building the model    
        xb = tf.random.normal([1]+list(self.inp_shape))
        indexb = tf.ones([1])
        self(xb, indexb)

    def call(self, x, index):
        index = get_timestep_embedding(index, self.c)
        index = tf.nn.swish(self.temb[0](index))
        index = self.temb[1](index)

        x = self.conv_in(x)
        residuals = []

        for block in self.down_layers:
            x = block(x, index)
            if not isinstance(block, downsample):
                residuals.append(x)
        
        for block in self.mid_layers:
            x = block(x, index)
        
        for block in self.up_layers:
            if not isinstance(block, upsample):
                x = tf.concat([x, residuals.pop()], axis=-1)
            x = block(x, index)
        
        x = tf.nn.swish(self.norm_out(x))
        return self.conv_out(x)
    
    def run_onestep(self, z):
        inp = tf.identity(z)
        index = tf.ones_like(z[:, 0, 0, 0]) * (self.n_steps - 1)
        x = self(z, index) 
        pred_y = inp - x
        return pred_y
    
    def run_ddim_step(self, xt, index, alpha, alpha_next):
        eps = self(xt, index)
        x_t_minus1 = tf.sqrt(alpha_next) * (xt - tf.sqrt(1-alpha)*eps) / tf.sqrt(alpha)
        x_t_minus1 += tf.sqrt(1-alpha_next) * eps 
        return x_t_minus1
    
    def run_ddpm_step(self, xt, index, beta, alpha):
        #argument alpha denotes alpha bar in the original paper
        #here uses 1 - beta for the regular alpha.
        eps = self(xt, index)
        eps = eps * beta / tf.sqrt(1 - alpha)
        sigma = tf.sqrt(beta)
        x_t_minus1 = (xt - eps)/tf.sqrt(1 - beta) + sigma * tf.random.normal(tf.shape(xt))
        return x_t_minus1
