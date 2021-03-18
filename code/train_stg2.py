from model_manager import ModelManager
import tensorflow as tf
from utils import to_numpy, save_samples
import numpy as np
import pickle
import os
from time import time

class Stage2Trainer(ModelManager):
    def __init__(self, stg2_data_dir, devices_to_use, model_dir, data_dir, results_dir, model_config, schedule_config, training_config, continue_training, use_mixed_precision, use_xla):
        super().__init__(devices_to_use, model_dir, data_dir, results_dir, model_config, schedule_config, use_mixed_precision, use_xla, show_mode='ONE', show_steps=None)
        self.stg2_data_dir = stg2_data_dir
        if continue_training:
            self.continue_num = self._get_last_ckpt_num(stg2=True)
        else:
            self.continue_num = None
        self._get_training_objects(**training_config)
        restored = self.try_restore_state()
        if not restored:
            self._start_new_train_run()
        self.moving_avg_weights = self.model.get_weights()
        self.i = int(to_numpy(self.optimizer.iterations))
        self.starttime = time()

    def _start_new_train_run(self):
        num = self._get_last_ckpt_num()
        self.model.load_weights(self._getp('ema_model_{}k.h5'.format(num)))
    
    def try_restore_state(self):
        num = self.continue_num

        if num is None:
            print("Starting new training session...")
            return False

        if not self.check_objects_exist(num, stg2=True):
            print("You are trying to continue training from iteration {}k but the necessary models/optimizers were not found.".format(num))
            print("We will instead start a new training session.")
            return False
        opt_path = self._getp('optimizer_stg2_{}k.p'.format(num))

        with open(opt_path, 'rb') as f:
            opt_weights = pickle.load(f)

        with self.strategy.scope():
            x = tf.random.normal([1] + list(self.model.inp_shape))
        self.train_step(x, x) #this will create the optimizer weights.
        self.load_models(num, stg2=True) 
        self.train_loss.reset_states()
        self.train_gradnorm.reset_states()
        self.optimizer.set_weights(opt_weights)
        print("successfully restored training state")
        return True
    
    @tf.function
    def train_step(self, x, y):
        def _step_fn(x, y):
            with tf.GradientTape() as tape:
                y_pred = self.model.run_onestep(x)
                loss = tf.reduce_mean(tf.square(y - y_pred))

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

        self.strategy.run(_step_fn, args=(x, y))

    def train(self):
        def preprocess_func(x, y):
            x = tf.cast(x, tf.float32)
            y = tf.cast(y, tf.float32)/127.5 - 1.
            return x, y

        logfile_path = os.path.join(self.results_dir, 'training_stg2_logfile.txt')
        if not os.path.exists(logfile_path):
            logfile = open(logfile_path, "w")

        lsdir_x = [f for f in os.listdir(self.stg2_data_dir) if 'data_x' in f]
        lsdir_y = [f for f in os.listdir(self.stg2_data_dir) if 'data_y' in f]
        lsdir_x, lsdir_y = sorted(lsdir_x), sorted(lsdir_y)
        assert len(lsdir_x) == len(lsdir_y), "You are missing an x or y dataset shard. You may have deleted it."
        while True:
            for xtr_path, ytr_path in zip(lsdir_x, lsdir_y):
                x_tr = np.load(os.path.join(self.stg2_data_dir, xtr_path))
                y_tr = np.load(os.path.join(self.stg2_data_dir, ytr_path))
                dataset = tf.data.Dataset.from_tensor_slices((x_tr, y_tr)).shuffle(self.batch_size*5).batch(self.batch_size, drop_remainder=True).map(preprocess_func)
                dataset = self.strategy.experimental_distribute_dataset(dataset)
                
                for x, y in dataset:
                    self.train_step(x, y)
                    
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
                        self.save_objects(i//1000, stg2=True)
                        _, samples_ema = self.generate_samples(64)
                        savepath = self._getp('samples_ema_{}k.jpg'.format(i//1000), folder='results')
                        save_samples(samples_ema, savepath)

                    if i >= self.max_iterations:
                        print("Training complete. Recover your models from the {} folder and your training results from the {} folder".format(self.model_dir, self.results_dir))
                        print("Training was done on a maximum of {} iterations".format(i))
                        return True

                    self.i += 1