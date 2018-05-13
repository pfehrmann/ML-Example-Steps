from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# define the CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')
parser.add_argument('--no_train', help='disable training', action='store_true')

# Constants for int labels
CAT_LABEL = 0
DOG_LABEL = 1

# the basepath of the training images. Make sure to have the data setup in the described way.
base_path = 'resized/data'

# Dimensions of the images. Preventing some magic numbers
image_size_rows = 112
image_size_cols = 112

# set this to 1 to have grayscale images. Then you will have to check how to display grayscale images.
image_size_channels = 3
image_size = [image_size_cols, image_size_rows, image_size_channels]


# load and decode the image from the disk.
def prepare_image_fn(image_path, label):
    img_file = tf.read_file(image_path)

    # TODO: This might fail from time to time. Take a close look at image 666.jpg. There are others like this :)
    # TODO: see https://www.tensorflow.org/api_docs/python/tf/contrib/data/ignore_errors, I din't try it...
    img = tf.image.decode_image(img_file, channels=image_size_channels)
    img.set_shape([None, None, None])

    # TODO: You may want to handle this externally
    img = tf.image.resize_images(img, [image_size_cols, image_size_rows])

    return img, label


# Get all images from a folder, assign a label to each of them.
def get_class_dataset(dataset, label, class_name):
    # list all images
    files = tf.data.Dataset.list_files(join(base_path, dataset, class_name, '*.jpg'))

    # TODO: We process 4 imahes in parallel, maybe use more, maybe less.
    images = files.map(lambda x: (x, label), 4)
    return images


def get_dataset(dataset='train', shuffle=True, batch_size=250, buffer_size=20000, repeat=True, prefetch=500):
    '''
    Create a dataset using the tensorflow.data.Dataset API.
    :param dataset: The name of the dataset, eg. train or test
    :param shuffle: Shuffle the data?
    :param batch_size: Return batches of this size.
    :param buffer_size: Buffer size for shuffle. Should be equal to dataset size.
    :param repeat: If true, repeats infinitely
    :param prefetch: Prefetch some data. Can be used to load new data while old data is processed on the GPU.
    :return: Returns a tf.data.Dataset of both cats and dogs
    '''
    cats = get_class_dataset(dataset, tf.constant(CAT_LABEL), 'Cat')
    dogs = get_class_dataset(dataset, tf.constant(DOG_LABEL), 'Dog')
    data = cats.concatenate(dogs)

    if repeat:
        data = data.repeat()

    if shuffle:
        data = data.shuffle(buffer_size=buffer_size)

    # use this, if you have a weaker gpu :(
    # TODO: We load 10 images parallel. That might be too much.
    data = data.map(prepare_image_fn, 10)
    data = data.batch(batch_size)

    # Use this, if you have a proper gpu :)
    # data = data.apply(tf.contrib.data.map_and_batch(
    #    map_func=prepare_image_fn, batch_size=batch_size, num_parallel_batches=1))

    if prefetch is not None:
        data = data.prefetch(prefetch)

    return data


# see https://www.tensorflow.org/tutorials/layers
# see https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/examples/tutorials/layers/cnn_mnist.py
def cnn_model_fn(features, labels, mode):
    pass
    # TODO: Define the model here.

def main(argv):
    args = parser.parse_args(argv[1:])
    # TODO: Use the model here.


if __name__ == '__main__':
    # TODO: if you want to train on your CPU for some reason (eg. not enough RAM)
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sess = tf.Session()
    with sess.as_default():
    #    tf.logging.set_verbosity(tf.logging.INFO)
    #    tf.app.run(main)

        # test pipeline
        dataset = get_dataset().make_one_shot_iterator()
        next_batch = dataset.get_next()
        image = next_batch[0].eval()
        image = np.resize(image, image.shape[-3:])
        image = image / 255
        plt.imshow(image)
        plt.show()
