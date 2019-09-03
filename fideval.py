# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)

# Various attempts at improving the WAE
import tensorflow as tf
import numpy as np
import wae
import os
import logging
from models import encoder, decoder
from datahandler import datashapes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import fid

import glob
from scipy.misc import imread

def generate(opts):
    MAX_GD_STEPS = 200
    LOSS_EVERY_STEPS = 50
    DEBUG = False
    NUM_POINTS = 10000
    BATCH_SIZE = 100

    checkpoint = os.path.join(opts['work_dir'], 'checkpoints', 'trained-wae-20000')

    net = wae.WAE(opts)

    net.saver.restore(net.sess, checkpoint)

    # Finally, start generating the samples

    res_samples = []

    for img_index in range(10):
        pics = net.sess.run(net.decoded,
            feed_dict={
                net.sample_noise: np.random.normal(size=(5, opts['zdim'])),
                net.is_training: False
            })

        res_samples.append(pics)

    samples = np.array(res_samples)
    samples = np.vstack(samples)
    pic_path = os.path.join(opts['work_dir'], 'checkpoints', 'dummy.samples%d' % (NUM_POINTS))
    np.save(pic_path, samples)
    for img_index, sample in enumerate(samples):
        #sample = cv2.cvtColor(sample, cv2.COLOR_GRAY2RGB)
        sample = sample / 2 + 0.5
        print(np.shape(sample))
        print(np.min(sample), np.max(sample))
        cv2.imwrite("results_celeba/generated/generated%03d.png" % img_index, sample * 255)


"""
# Paths
image_path = 'results_celeba/generated' # set path to some generated images
stats_path = 'fid_stats_celeba.npz' # training set statistics
inception_path = fid.check_or_download_inception(None) # download inception network

# load precalculated training set statistics
f = np.load(stats_path)
mu_real, sigma_real = f['mu'][:], f['sigma'][:]
f.close()

image_list = glob.glob(os.path.join(image_path, '*.png'))
images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])

fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mu_gen, sigma_gen = fid.calculate_activation_statistics(images, sess)

fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
print("FID: %s" % fid_value)
"""