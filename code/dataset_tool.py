import tensorflow as tf
import numpy as np
import os
import imageio
from utils import get_shardsize, get_zeros_array

def resize(image, target_shape):
    imdtype = image.dtype
    with tf.device('/CPU:0'):
        image = tf.image.resize(image, target_shape[:2]).numpy()
    assert image.shape == target_shape
    return image.astype(imdtype)

def fit_image(image, target_shape, fit_method):
    assert isinstance(image, np.ndarray), "Image must be numpy array"
    assert len(image.shape) == 3 and image.shape[-1] == 3, "Original Image shape must be of shape (H, W, 3)."
    assert len(target_shape) == 3 and target_shape[-1] == 3, "Desired Image shape must be of shape (H, W, 3)."
    assert fit_method in ['resize', 'center_crop', 'random_crop'], "Crop method must be one of 'resize', 'center_crop' or 'random_crop' "

    (h, w, _), (htar, wtar, _) = image.shape, target_shape

    if image.shape == target_shape:
        return image
    
    if h < htar or w < wtar:
        if fit_method != 'resize':
            print("Your selected fit method is {} but your desired image shape is larger than the given image's shape. Using resize instead - note that this may change the image aspect ratio.".format(fit_method), end="\r")
        return resize(image, target_shape)
    
    if fit_method == 'resize':
        return resize(image, target_shape)

    elif fit_method == 'center_crop':
        trim_h = int((h - htar)/2)
        trim_w = int((w - wtar)/2)

        image = image[trim_h:h-trim_h, trim_w:w-trim_w]
        if image.shape[0] != htar:
            image = image[:-1]
        if image.shape[1] != wtar:
            image = image[:, :-1]
        assert image.shape == target_shape, image.shape
        return image

    elif fit_method == 'random_crop':
        imdtype = image.dtype
        with tf.device('/CPU:0'):
            image = tf.image.random_crop(tf.constant(image), target_shape).numpy()
        assert image.shape == target_shape
        return image.astype(imdtype)
        
    
def images_to_train_dataset(writedir, datadir, target_shape, fit_method='resize'):
    '''
    writedir: specifies the folder where numpy arrays are created
    datadir: specifies the folder where the jpg/png files in the dataset are located
    target_shape: the desired shape of the images
    fit_method: how to adjust images such that they are of the same shape as target_shape. must be 'resize', 'center_crop' or 'random_crop'
    remove_datadir: whether or not to delete the original dataset

    returns: the number of training examples.
    '''
    
    if len(os.listdir(datadir)) == 0:
        raise RuntimeError("No training images were found. Data directory should not be empty. ")
    elif os.path.isfile(datadir):
        raise RuntimeError("data directory should not be a file, it should be a folder. You may have to unzip your files to a new folder.")

    if os.path.isfile(writedir):
        raise RuntimeError("The directory you want to write to is an existing file.")
    elif writedir == datadir:
        raise RuntimeError("The numpy arrays should be written to a different directory than the original.")
    elif os.path.isdir(writedir):
        if len(os.listdir(writedir)) != 0:
            print("Files already exist in this directory. Will use these for training.")
            return len(os.listdir(writedir)) 
    else:
        os.mkdir(writedir)

    shard_size = get_shardsize(target_shape)
    numpy_dataset = get_zeros_array(target_shape)
    tmp_numpy = get_zeros_array(target_shape) #appends to numpy_dataset in groups of size 50. this is faster.

    count = 0
    files_written = 0

    for impath in sorted(os.listdir(datadir)):
        impath = os.path.join(datadir, impath)
        try:
            image = imageio.imread(impath)
        except:
            continue #cant be converted to numpy array.
        image = fit_image(image, target_shape, fit_method)
        assert len(image.shape) == 3
        image = np.expand_dims(image, axis=0)
        count += 1

        tmp_numpy = np.concatenate((tmp_numpy, image), axis=0)
        if tmp_numpy.shape[0]%64 == 0:
            numpy_dataset = np.concatenate((numpy_dataset, tmp_numpy))
            tmp_numpy = get_zeros_array(target_shape)

        if numpy_dataset.shape[0] >= shard_size:
            data_to_write, remaining_data = numpy_dataset[:shard_size], numpy_dataset[shard_size:]
            print(data_to_write.shape, remaining_data.shape)
            writepath = os.path.join(writedir, 'data_{}.npy'.format(files_written))
            np.save(writepath, data_to_write)
            files_written += 1
            numpy_dataset = remaining_data

    numpy_dataset = np.concatenate((numpy_dataset, tmp_numpy))
        
    writepath = os.path.join(writedir, 'data_{}.npy'.format(files_written))
    if numpy_dataset.shape[0] != 0:
        np.save(writepath, numpy_dataset)
        files_written += 1

    print("A maximum of %d images will be used in training." % count)

    return count

