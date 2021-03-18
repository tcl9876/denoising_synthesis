import os
import numpy as np
import tensorflow as tf
from utils import to_numpy, get_betas, LinearWarmup
import pickle
from models import GenerativeModel
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from time import time

class ModelManager(object):
    def __init__(self, devices_to_use, model_dir, data_dir, results_dir, model_config, schedule_config, use_mixed_precision, use_xla, show_mode, show_steps):

        self.use_mixed_precision = use_mixed_precision
        if use_mixed_precision:
            mixed_precision.set_policy('mixed_float16')
        tf.config.optimizer.set_jit(bool(use_xla))

        strategy, machine_type = self._get_strategy(devices_to_use)

        self.strategy = strategy
        self.machine_type = machine_type

        with self.strategy.scope():   
            self.model = GenerativeModel(**model_config)
            self.ema_model = GenerativeModel(**model_config)

        self.beta_set = get_betas(**schedule_config)
        self.alpha_set = tf.math.cumprod(1 - self.beta_set)
        self.use_mixed_precision = use_mixed_precision
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.results_dir = results_dir   
        
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError("Your provided data directory does not exist.")

        assert show_mode.upper() in ['DDIM', 'DDPM', 'ONE'], """show_mode must be in ['DDIM', 'DDPM', 'ONE']. 
            'DDIM' denotes producing samples with a DDIM, DDPM denotes producing samples with a DDPM,
            ONE denotes using a one-step generative model. If unsure, leave as the default ('DDIM') """
        self.show_mode = show_mode.upper()
        
        if show_steps is not None:
            assert isinstance(show_steps, int) and 2 <= show_steps <= len(self.beta_set), "show_steps must be either None or an integer in [2, T]. If unsure, leave as the default (None)"
            if show_steps < len(self.beta_set) and self.show_mode == 'DDPM':
                print("Provided show_steps argument incompatible with 'DDPM' mode. setting show_steps to {}.".format(len(self.beta_set)))
                show_steps = len(self.beta_set)
        
        self.show_steps = len(self.beta_set) if show_steps is None else show_steps

    def _getp(self, name, folder='models'):
        if folder=='models':
            dr = self.model_dir
        elif folder=='results':
            dr = self.results_dir
        return os.path.join(dr, name)

    def check_objects_exist(self, num, stg2=False):
        s2 = '_stg2' if stg2 else ''
        model_path = self._getp('nonema_model{}_{}k.h5'.format(s2, num))
        ema_model_path = self._getp('ema_model{}_{}k.h5'.format(s2, num))
        opt_path = self._getp('optimizer{}_{}k.p'.format(s2, num))
        return os.path.exists(model_path) and os.path.exists(ema_model_path) and os.path.exists(opt_path)
            
    def _get_last_ckpt_num(self, stg2=False):    
        def stg2_req(f):
            if stg2:
                return 'stg2' in f
            else:
                return 'stg2' not in f
    
        npaths = [f for f in os.listdir(self.model_dir) if ('nonema_model' in f and stg2_req(f))]
        if not npaths:
            return None
        
        numbers = []
        for nema_path in npaths:
            st = nema_path[7:].index("_")
            end = nema_path.index("k.h5")
            if stg2:
                numbers.append(int(nema_path[st+13:end]))
            else:
                numbers.append(int(nema_path[st+8:end]))
    
        return sorted(numbers)[-1]
    
    def save_objects(self, num, stg2=False):
        s2 = '_stg2' if stg2 else ''
        model_path = self._getp('nonema_model{}_{}k.h5'.format(s2, num))
        ema_model_path = self._getp('ema_model{}_{}k.h5'.format(s2, num))
        opt_path = self._getp('optimizer{}_{}k.p'.format(s2, num))

        self.model.save_weights(model_path)
        self.ema_model.set_weights(self.moving_avg_weights)
        self.ema_model.save_weights(ema_model_path)
        with open(opt_path, 'wb') as f:
            pickle.dump(self.optimizer.get_weights(), f, pickle.HIGHEST_PROTOCOL)
    
    def load_models(self, num, stg2=False):
        s2 = '_stg2' if stg2 else ''
        model_path = self._getp('nonema_model{}_{}k.h5'.format(s2, num))
        ema_model_path = self._getp('ema_model{}_{}k.h5'.format(s2, num))
        if os.path.exists(model_path) and os.path.exists(ema_model_path):
            self.model.load_weights(model_path)
            self.ema_model.load_weights(ema_model_path)
            self.moving_avg_list = self.ema_model.get_weights()
        else:
            raise FileNotFoundError('Could not find the files to load the objects.')

    def _get_strategy(self, devices_to_use=None):
        if isinstance(devices_to_use, str):
            if "CPU" in devices_to_use.upper():
                return tf.distribute.get_strategy(), "CPU"
        
        if tf.config.list_physical_devices('GPU'):
            return tf.distribute.MirroredStrategy(devices=devices_to_use), "GPU"
        else:
            return tf.distribute.get_strategy(), "CPU"

    def _make_runfn(self, model, strategy, op_type):
        if op_type is None:
            op_type = 'Onestep'

        assert op_type in ['DDIM', 'DDPM', 'ONE']
        if op_type == 'ONE':
            @tf.function
            def runmodel(z):
                def replica_fn(z):
                    return model.run_onestep(z)
                return self.strategy.run(replica_fn, args=(z,))
        elif op_type == 'DDIM':
            @tf.function
            def runmodel(xt, index, alpha, alpha_next):
                def replica_fn(xt, index, alpha, alpha_next):
                    return model.run_ddim_step(xt, index, alpha, alpha_next)
                return self.strategy.run(replica_fn, args=(xt, index, alpha, alpha_next))
        elif op_type == 'DDPM':
            @tf.function
            def runmodel(xt, index, beta, alpha):
                def replica_fn(xt, index, beta, alpha): 
                    return model.run_ddpm_step(xt, index, beta, alpha)
                return self.strategy.run(replica_fn, args=(xt, index, beta, alpha))

        return runmodel

    def generate_samples(self, n_samples, batch_size=None, verbose=True):
        model = self.ema_model

        def randn_func(xtr):
            return tf.random.normal([tf.shape(xtr)[0], self.model.h, self.model.w, 3], dtype=tf.float32)

        seqlen = len(self.beta_set)
        if self.show_mode == 'ONE':
            seq = [0]
            get_xtm1 = self._make_runfn(model, self.strategy, 'ONE')
        elif self.show_mode == 'DDIM':
            seq = range(0, seqlen,  seqlen//self.show_steps)
            get_xtm1 = self._make_runfn(model, self.strategy, 'DDIM')
        else:
            seq = range(seqlen)
            get_xtm1 = self._make_runfn(model, self.strategy, 'DDPM')

         
        seq_next = [-1] + list(seq[:-1])
        if not isinstance(batch_size, int):
            sampling_bs = min(self.batch_size, 64)
        else:
            sampling_bs = batch_size
        
        assert n_samples%sampling_bs == 0
            
        noise_data = tf.data.Dataset.range(n_samples).batch(sampling_bs, drop_remainder=True).map(randn_func)
        noise_data = self.strategy.experimental_distribute_dataset(noise_data)
        inps = np.zeros([0, self.model.h, self.model.w, 3]).astype('float16')
        outs = np.zeros([0, self.model.h, self.model.w, 3]).astype('uint8')
        s = time()
        for x in noise_data:

            bs = to_numpy(x).shape[0]//self.strategy.num_replicas_in_sync #the per-replica batch size on inference mode.
            
            z = tf.cast(tf.identity(x), tf.float16)
            for i, j in zip(reversed(seq), reversed(seq_next)): 
                index = tf.constant(i, dtype=tf.float32) * tf.ones([bs])

                alpha = self.alpha_set[i] * tf.ones([bs, 1, 1, 1]) 

                alpha_next = self.alpha_set[j] if j>=0 else tf.constant(1.0)
                alpha_next = alpha_next * tf.ones([bs, 1, 1, 1]) 
                beta = self.beta_set[i] * tf.ones([bs, 1, 1, 1]) 

                if self.show_mode == 'DDIM':
                    x = get_xtm1(x, index, alpha, alpha_next)
                elif self.show_mode=='DDPM':
                    x = get_xtm1(x, index, beta, alpha)
                else:
                    x = get_xtm1(x)
        
            x = np.clip(to_numpy(x), -1.0, 1.0)
            x = (x+1.)*127.5
            outs = np.concatenate((outs, x.astype('uint8')), axis=0)
            inps = np.concatenate((inps, to_numpy(z)), axis=0)
        
        if verbose:    
            print("Generated %d samples. Time taken: %0.2f sec" % (outs.shape[0], time()-s))
        assert inps.shape == outs.shape
        return inps, outs
    
    #TRAINING RELATED FUNCTIONS THAT BOTH WILL USE.
    def _update_moving_avg(self): #TAKES A LONG TIME, SO IS NOT DONE ON EVERY STEP.
        self.moving_avg_weights = [(m * self.ema_rate + x * (1 - self.ema_rate)) for m, x in zip(self.moving_avg_weights, self.model.get_weights())]

    def _get_training_objects(self, lr, beta_1, beta_2, ema_rate, max_iterations, batch_size):

        assert isinstance(lr[0], float)  and lr[0] > 0, "the first argument to lr should be a float representing the maximum learning rate."
        assert isinstance(lr[1], int) and lr[1] >= 0, "the second argument to lr should be an integer representing the number of warmup steps. for no warmup at all use 0."
        self.lr = LinearWarmup(lr[0], lr[1])

        assert 0 <= beta_1 < 1 and isinstance(beta_1, float), "beta_1 must be float in [0, 1)"
        assert 0 <= beta_2 < 1 and isinstance(beta_2, float), "beta_2 must be float in [0, 1)"
        assert 0 <= ema_rate < 1 and isinstance(ema_rate, float), "ema_rate must be float in [0, 1). Use 0 for no moving average, although this is not recommended."
        self.ema_rate = ema_rate

        with self.strategy.scope():
            self.optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=beta_1, beta_2=beta_2)
            if self.use_mixed_precision:
                self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer, loss_scale='dynamic')

        assert isinstance(max_iterations, int), "iterations must be an integer"
        self.max_iterations = max_iterations
        assert isinstance(batch_size, int) and batch_size > 1, "batch_size must be an integer greater than 1."
        self.batch_size = batch_size

        with self.strategy.scope():
            self.train_loss = tf.keras.metrics.Mean()
            self.train_gradnorm = tf.keras.metrics.Mean()
