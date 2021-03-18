from model_manager import ModelManager
import os
import numpy as np
import tensorflow as tf

def write_stg2_files(write_dir, samples_to_write, shard_size, batch_size, devices_to_use, model_dir, data_dir, results_dir, model_config, schedule_config, use_mixed_precision, use_xla, show_steps):
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)
    elif os.path.isdir(write_dir):
        if len(os.listdir(write_dir)) != 0:
            print("The stage 2 data directory is not empty, so training will use the files in here.")
            return 
    show_mode = 'DDIM'
    manager = ModelManager(devices_to_use, model_dir, data_dir, results_dir, model_config, schedule_config, use_mixed_precision, use_xla, show_mode, show_steps)
    num = manager._get_last_ckpt_num()
    manager.load_models(num)
    n_shards = samples_to_write//shard_size
    remainder = samples_to_write%shard_size
    h, w = manager.ema_model.h, manager.ema_model.w
    if remainder != 0:
        n_shards += 1
    n_total_ex = 0
    for i in range(n_shards):
        data_x = np.zeros((0, h, w, 3)).astype('float16')
        data_y = np.zeros((0, h, w, 3)).astype('uint8')
        if i == n_shards - 1 and remainder != 0:
            ss = remainder
        else:
            ss = shard_size
        shard_rem = ss%batch_size
        for j in range(ss//batch_size):
            inps, outs = manager.generate_samples(batch_size, batch_size, verbose=False)
            data_x = np.concatenate((data_x, inps))
            data_y = np.concatenate((data_y, outs))
        if shard_rem != 0:
            inps, outs = manager.generate_samples(shard_rem, shard_rem, verbose=False)
            data_x = np.concatenate((data_x, inps))
            data_y = np.concatenate((data_y, outs))
        
        assert data_x.shape[0] == ss and data_x.shape == data_y.shape
        x_savepath = os.path.join(write_dir, 'data_x_{}'.format(i))
        y_savepath = os.path.join(write_dir, 'data_y_{}'.format(i))
        np.save(x_savepath, data_x)
        np.save(y_savepath, data_y)
        n_total_ex += len(data_x)
        print("Finished writing {} examples to {}".format(n_total_ex, write_dir))