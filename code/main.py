import os
import numpy as np
import tensorflow as tf 
from time import time
import yaml
import random

if __name__ == '__main__':
    
    tf.random.set_seed(123)
    np.random.seed(123)
    random.seed(123)

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('option', type=str, help="whether to train a model or evaluate using an existing one. Must be either 'train' or 'evaluate'")
    parser.add_argument('--data_loc', type=str, help="where your dataset directory is located. Required argument if training. ")
    parser.add_argument('--yaml_loc', type=str, default='config.yaml', help="where your yaml config file is located.")
    parser.add_argument('--figure_path', type=str, help="The path to save the figure of generated images created during evaluation.")
    parser.add_argument('--eval_examples', type=int, default=16, help="How many examples to create during evaluation. defaults to 16.")
    parser.add_argument('--no_stg2', action='store_true', help="Whether to evaluate or train without the stage 2 models. defaults to False")
    args = parser.parse_args()
    
    with open(args.yaml_loc, 'r') as f:
        config = yaml.safe_load(f)

    target_shape = tuple(config['target_shape'])
    continue_training = config['continue_training']
    fit_method  = config['fit_method']
    data_dir = config['data_dir']
    model_dir = config['model_dir']
    results_dir = config['results_dir']
    data_stg2_dir = config['data_stg2_dir']
    results_stg2_dir = config['results_stg2_dir']
    devices_to_use = config['devices_to_use']
    use_mixed_precision = config['use_mixed_precision']
    use_xla = config['use_xla']
    show_steps = config['show_steps']
    show_mode  = config['show_mode']
    model_config = config['model_config']
    schedule_config = config['schedule_config']
    train_base_config  = config['train_base_config']
    dataset_stg2_config = config['dataset_stg2_config']
    train_stg2_config  = config['train_stg2_config']
    
    model_config['n_steps'] = schedule_config['n_steps']
    model_config['input_shape'] = target_shape
    

    if args.option == 'train':
        #create the images dataset
        from dataset_tool import images_to_train_dataset
        
        original_data_dir = args.data_loc
        if not isinstance(original_data_dir, str):
            print("You have not provided the location of the dataset. Use --data_loc {LOCATION OF YOUR DATA} unless you are continuing training and numpy files have been created already.")
        else:
            images_to_train_dataset(data_dir, original_data_dir, target_shape, fit_method)
        
        #train the model
        from train_base import BaseTrainer
        trainer = BaseTrainer(devices_to_use, model_dir, data_dir, results_dir, model_config,
                      schedule_config, train_base_config, continue_training,
                      use_mixed_precision, use_xla, show_mode, show_steps)
        trainer.train()
        
        
        #optionally run the second stage of the model.
        if not args.no_stg2:

            #creates the synthetic dataset used by the 2nd model
            from dataset_stg2 import write_stg2_files
            from utils import get_shardsize
            samples_to_write = dataset_stg2_config['samples_to_write']
            dataset_show_steps = dataset_stg2_config['show_steps']
            batch_size = dataset_stg2_config['batch_size']
            shard_size = get_shardsize(target_shape)//2
            
            write_stg2_files(data_stg2_dir, samples_to_write, shard_size, batch_size,
                 devices_to_use, model_dir, data_dir, results_dir, model_config,
                schedule_config, use_mixed_precision, use_xla, dataset_show_steps)

            #trains the 2nd stage of the model
            from train_stg2 import Stage2Trainer
            stg2_trainer = Stage2Trainer(data_stg2_dir, devices_to_use, model_dir, data_dir,
                              results_stg2_dir, model_config, schedule_config, train_stg2_config,
                              continue_training, use_mixed_precision, use_xla)
            stg2_trainer.train()

        print("Training is complete.")

    elif args.option == 'eval':
        stg2 = not bool(args.no_stg2)
        if stg2:
            show_mode = 'ONE'
        print("SHOW MODE: {}".format(show_mode))
        from model_manager import ModelManager
        evaluator = ModelManager(devices_to_use, model_dir, data_dir, results_dir, model_config,
         schedule_config, use_mixed_precision, use_xla, show_mode, show_steps)
        
        num = evaluator._get_last_ckpt_num(stg2=stg2)
        print("Restoring model {}k".format(num))
        evaluator.load_models(num, stg2=stg2)
        

        n_ex = args.eval_examples

        if n_ex > 64:
            print("Evaluation is only supported for 64 or fewer examples. Reducing the number of examples to generate to 64...")
            n_ex = 64
        _, outputs = evaluator.generate_samples(n_ex, n_ex, verbose=False)

        from utils import save_samples
        save_path = args.figure_path
        save_samples(outputs, save_path)        
        print("Evaluation is complete. Samples have been saved to {}.".format(save_path))

    else:
        raise ValueError("When running from command line, the option argument should be either 'train' or 'eval'.")
