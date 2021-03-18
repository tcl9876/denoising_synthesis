
import tensorflow as tf
import numpy as np
from tensorflow.python.distribute.values import PerReplica
import matplotlib.pyplot as plt

class LinearWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, learnrate, warmup_steps):
        super().__init__()
        self.learnrate = learnrate
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = self.learnrate
        arg2 = step * self.learnrate / float(self.warmup_steps)
        return tf.math.minimum(arg1, arg2)

def get_random_inds_betas_alphas(batch_size, beta_set, alpha_set):
    indices = tf.random.uniform([batch_size, 1], maxval=len(beta_set), dtype=tf.int32)
    batch_size = tf.shape(indices)[0]
    betas = tf.gather_nd(beta_set, indices)
    alphas = tf.gather_nd(alpha_set, indices)
    indices = tf.reshape(indices, [batch_size])
    betas = tf.reshape(betas, [batch_size, 1, 1, 1])
    alphas = tf.reshape(alphas, [batch_size, 1, 1, 1])
    return indices, betas, alphas


def get_betas(n_steps, betas_type="linear", min=None, max=None, const=None):
    assert isinstance(n_steps, int) and n_steps > 1, "The number of steps used must be an integer greater than 1" 
    if betas_type == 'linear':
        if max is None or min is None:
            raise ValueError("must specify max and min values when using betas type of 'linear' ")
        else:
            assert max > 0 and max < 1 and min >= 0 and min < 1 and max > min, "min and max must be in (0, 1) and max > min"
        return tf.linspace(min, max, n_steps)
    elif betas_type == 'quadratic':
        if max is None or min is None:
            raise ValueError("must specify max and min values when using betas type of 'quadratic' ")
        else:
            assert max > 0 and max < 1 and min >= 0 and min < 1 and max > min, "min and max must be in (0, 1) and max > min"
        return tf.square(tf.linspace(tf.sqrt(min), tf.sqrt(max), n_steps))
    elif betas_type == 'constant':
        if const is None:
            raise ValueError("must specify const value when using betas type of 'constant' ")
        else:
            assert const > 0 and const < 1, "const must be in (0, 1)"
        return tf.square(tf.linspace(const, const, n_steps))
    else:
        raise NotImplementedError("betas_type should be one of ['linear', 'quadratic', 'constant']")

def to_numpy(x):
    if isinstance(x, PerReplica): #per replica object only (like ones from a experimental_distribute_dataset)
        return tf.concat(x.values, axis=0)
    else:
        try:
            x = x.numpy()
            return x
        except:
            assert isinstance(x, np.ndarray), "Not a valid value provided. The type of the value provided was: {}".format(str(type(x)))
            return x

def save_samples(x, savepath):
    assert isinstance(x, np.ndarray) and isinstance(savepath, str)
    assert x.dtype == 'uint8' or x.dtype == np.uint8
    if len(x.shape)==3:
        x = np.expand_dims(x, axis=0)
    assert len(x.shape) == 4 and x.shape[-1] == 3


    nimg = len(x)
    nimgrt = np.ceil(np.sqrt(nimg))
    figh, figw = nimgrt*x.shape[1]//48, nimgrt*x.shape[2]//48
    fig = plt.figure(figsize=(figh, figw))
    
    for i in range(nimg):
        plt.subplot(nimgrt, nimgrt, i+1)
        plt.imshow((x[i]))
        plt.axis('off')

    plt.savefig(savepath)

def get_shardsize(target_shape):
    shardsize = (1024 * 1024 * 1536) / np.prod(np.array(target_shape))
    return int(shardsize)

def get_zeros_array(target_shape):
    return np.zeros((0,) +  tuple(target_shape)).astype('uint8')
