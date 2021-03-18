from model_manager import ModelManager
import tensorflow as tf
from utils import to_numpy, get_random_inds_betas_alphas, save_samples
import numpy as np
import pickle
import os
from time import time

class BaseTrainer(ModelManager):
    def __init__(self, devices_to_use, model_dir, data_dir, results_dir, model_config, schedule_config, training_config, continue_training, use_mixed_precision, use_xla, show_mode, show_steps=None):
        super().__init__(devices_to_use, model_dir, data_dir, results_dir, model_config, schedule_config, use_mixed_precision, use_xla, show_mode, show_steps)
        if continue_training:
            self.continue_num = self._get_last_ckpt_num()
        else:
            self.continue_num = None
        self._get_training_objects(**training_config)
        restored = self.try_restore_state()
        if not restored:    
            self.moving_avg_weights = self.model.get_weights()
        else:
            self.moving_avg_weights = self.ema_model.get_weights()
        self.i = int(to_numpy(self.optimizer.iterations))
        self.starttime = time()
        
    @tf.function
    def train_step(self, x):
        def _step_fn(x):
            indices, betas, alphas = get_random_inds_betas_alphas(tf.shape(x)[0], self.beta_set, self.alpha_set)
            eps = tf.random.normal(tf.shape(x))
            x_perturbed = tf.sqrt(alphas) * x 
            x_perturbed += tf.sqrt(1 - alphas) * eps
            
            with tf.GradientTape() as tape:
                eps_pred = self.model(x_perturbed, indices, training=True)
                loss = tf.reduce_mean(tf.square(eps - eps_pred))

                if self.use_mixed_precision:
                    scaled_loss = self.optimizer.get_scaled_loss(loss)
                
            if self.use_mixed_precision:
                scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_variables) 
                gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
            else:
                gradients = tape.gradient(loss, self.model.trainable_variables) 

            self.train_loss(loss)
            self.train_gradnorm(tf.linalg.global_norm(gradients))

            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
        self.strategy.run(_step_fn, args=(x,))

    def try_restore_state(self):
        num = self.continue_num 

        if num is None:
            print("Starting new training session...")
            return False
        if not self.check_objects_exist(num):
            print("You are trying to continue training from iteration {}k but the necessary models/optimizers were not found.".format(num))
            print("We will instead start a new training session.")
            return False

        opt_path = self._getp('optimizer_{}k.p'.format(num))

        with open(opt_path, 'rb') as f:
            opt_weights = pickle.load(f)

        with self.strategy.scope():   
            x = tf.random.normal([1] + list(self.model.inp_shape))
        self.train_step(x) #this will create the optimizer weights.
        self.load_models(num, stg2=False) 
        self.train_loss.reset_states()
        self.train_gradnorm.reset_states()
        self.optimizer.set_weights(opt_weights)
        print("successfully restored training state")
        return True
            
    def train(self):
        def preprocess_func(x):
            return tf.cast(x, tf.float32)/127.5 - 1.

        logfile_path = os.path.join(self.results_dir, 'training_base_logfile.txt')
        if not os.path.exists(logfile_path):
            logfile = open(logfile_path, "w")

        while True:
            for dataset_path in sorted(os.listdir(self.data_dir)):
                dataset = np.load(os.path.join(self.data_dir, dataset_path))
                dataset = tf.data.Dataset.from_tensor_slices((dataset)).shuffle(self.batch_size*5).batch(self.batch_size, drop_remainder=True).map(preprocess_func)
                dataset = self.strategy.experimental_distribute_dataset(dataset)
                
                for x in dataset:
                    self.train_step(x)
                    
                    i = self.i
                    
                    if i%25 == 0: #updates the moving average every 25 steps , as it takes a very long time.
                        self._update_moving_avg()
                    if i%1000==0:
                        loss, gradnorm = to_numpy(self.train_loss.result()), to_numpy(self.train_gradnorm.result())
                        result_str = "Iteration: %d, Time Elapsed: %0.1f,  Loss: %0.4f, gradnorm: %0.4f" % (i, time()-self.starttime, loss, gradnorm)
                        print(result_str)
                        logfile = open(logfile_path, "a")
                        logfile.write(result_str + "\n")
                        logfile.close()
                        self.train_loss.reset_states()
                        self.train_gradnorm.reset_states()
                    if i%10000==0:
                        self.ema_model.set_weights(self.moving_avg_weights)
                        self.save_objects(i//1000)
                        _, samples_ema = self.generate_samples(64)
                        savepath = self._getp('samples_ema_{}k.jpg'.format(i//1000), folder='results')
                        save_samples(samples_ema, savepath)

                    if i >= self.max_iterations:
                        print("Training complete. Recover your models from the {} folder and your training results from the {} folder".format(self.model_dir, self.results_dir))
                        print("Training was done on a maximum of {} iterations".format(i))
                        return True

                    self.i += 1