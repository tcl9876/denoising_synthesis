-- The denoising synthesis module --

Usage:
To train a new model:
`python main.py train --data_loc (DATA_LOC)`

where DATA_LOC contains images (.jpg, .png, etc) to be used to train the model

To generate samples from a trained model do:  
`python main.py eval --figure_path (FIGURE_PATH)`

where FIGURE_PATH is the desired path for the output images.
you can also change the number of examples to generate and save using the --eval_examples argument

you can edit the config.yaml file to change a variety of hyperparamters including:
model width, model depth, image size, training steps, learning rate, batch size, and others.

If you create a new config file, make sure to specify its location using the --yaml_loc argument 

The code includes different variants of denoising-based models, including:
DDPMs or denoising diffusion probabilistic models, DDIMs or denoising diffusion implicit models, and a Denoising Student. 
Both DDPMs and DDIMs are iterative, meaning they use many iterations to produce a sample. They both reverse a noise adding "diffusion process" that is defined by a sequence of beta values. The main difference is that a DDPM adds a decreasing amount of noise to the data at each step of generation, while a DDIM does not. As a result, the DDIM process is deterministic given the same input noise, while the DDPM is not. For more information on DDPMs and DDIMs see https://arxiv.org/abs/2006.11239 and https://arxiv.org/abs/2010.02502.

Finally, the Denoising Student is a model that learns the same noise-to-data mapping as a DDIM model, only in one step as opposed to multiple. To implement this, we sample noise vectors from a standard normal, then give these to our trained DDIM which results in images. We save the noise and its corresponding images and train a second neural network to produce the same image given the  noise. This results in faster sampling speed, at a cost of image quality. For more information see https://arxiv.org/abs/2101.02388. If you want to only use a DDPM or DDIM, use the --no_stg2 argument and specify the appropriate model type in the config file.