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

def generate(opts):
    MAX_GD_STEPS = 200
    LOSS_EVERY_STEPS = 50
    DEBUG = False
    NUM_POINTS = 10000
    BATCH_SIZE = 100

    checkpoint = os.path.join(opts['work_dir'], 'checkpoints', 'trained-wae-final-50')

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
        sample = cv2.cvtColor(sample, cv2.COLOR_GRAY2RGB)
        print(np.shape(sample))
        print(np.min(sample), np.max(sample))
        cv2.imwrite("results_mnist/generated/generated%03d.png" % img_index, sample * 255)
