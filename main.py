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
base_path = 'data'


def main(argv):
    args = parser.parse_args(argv[1:])


if __name__ == '__main__':
    # TODO: if you want to train on your CPU for some reason (eg. not enough RAM)
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sess = tf.Session()
    with sess.as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        tf.app.run(main)
