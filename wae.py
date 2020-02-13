
# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)

""" Wasserstein Auto-Encoder models

"""

import sys
import time
import os
import numpy as np
import tensorflow as tf
import logging
import ops
import utils
import sinkhorn
import sparsifiers
import random
import neptune
import PIL
from models import encoder, decoder, z_adversary
from datahandler import datashapes
import improved_wae
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import plot_syn
from collections import OrderedDict
from scipy.stats import norm

class WAE(object):

    def __init__(self, opts, train_size=0):

        logging.error('Building the Tensorflow Graph')

        logging.info('Setting seed to: ' + str(opts['seed']))
        tf.set_random_seed(opts['seed'])
        np.random.seed(opts['seed'])

        self.sess = tf.Session()
        self.opts = opts
        self.train_size = train_size
        self.nat_pos = 0

        self.tensors_to_log = OrderedDict()

        # -- Some of the parameters for future use

        assert opts['dataset'] in datashapes, 'Unknown dataset.'
        self.data_shape = datashapes[opts['dataset']]

        # -- Placeholders

        self.add_inputs_placeholders()

        self.add_training_placeholders()
        sample_size = tf.shape(self.sample_points)[0]
        self.sample_size = sample_size

        self.add_nat_placeholders()

        # -- Transformation ops

        # Encode the content of sample_points placeholder
        res = encoder(opts, inputs=self.sample_points,
                      is_training=self.is_training)
        if opts['e_noise'] in ('deterministic', 'implicit', 'add_noise'):
            self.enc_mean, self.enc_sigmas = None, None
            if opts['e_noise'] == 'implicit':
                self.encoded, self.encoder_A = res
            else:
                self.encoded, _ = res
        elif opts['e_noise'] == 'gaussian':
            # Encoder outputs means and variances of Gaussian
            enc_mean, enc_sigmas = res[0]
            enc_sigmas = tf.clip_by_value(enc_sigmas, -50, 50)
            self.enc_mean, self.enc_sigmas = enc_mean, enc_sigmas
            if opts['verbose']:
                self.add_sigmas_debug()

            eps = tf.random_normal((sample_size, opts['zdim']),
                                   0., 1., dtype=tf.float32)
            self.encoded = self.enc_mean + tf.multiply(
                eps, tf.sqrt(1e-8 + tf.exp(self.enc_sigmas)))
            # self.encoded = self.enc_mean + tf.multiply(
            #     eps, tf.exp(self.enc_sigmas / 2.))

        # Decode the points encoded above (i.e. reconstruct)
        self.reconstructed, self.reconstructed_logits = \
                decoder(opts, noise=self.encoded,
                        is_training=self.is_training)

        # Decode the content of sample_noise
        self.decoded, self.decoded_logits = \
            decoder(opts, reuse=True, noise=self.sample_noise,
                    is_training=self.is_training)

        # -- Objectives, losses, penalties

        self.add_nat_tensors()
        self.zxz_loss = self.zxz_loss()
        self.penalty, self.loss_gan = self.matching_penalty()
        self.stay_loss = self.stay_loss()
        self.loss_reconstruct, self.per_sample_rec_loss = self.reconstruction_loss(
            self.opts, self.sample_points, self.reconstructed)
        self.latents_ph = tf.placeholder(tf.float32, shape=(opts['nat_size'], opts['zdim']))
        self.targets_ph = tf.placeholder(tf.float32, shape=(opts['nat_size'], opts['zdim']))
        self.sinkhorn_loss_tf = self.sinkhorn_loss(self.latents_ph, self.targets_ph)

        self.rec_grad = tf.reduce_mean(tf.abs(tf.gradients(self.loss_reconstruct, [self.encoded])))

        if self.stay_loss == None:
            self.wae_objective = self.rec_lambda * self.loss_reconstruct + \
                self.wae_lambda * self.penalty
        else:
            stay_lambda = opts['stay_lambda']
            self.wae_objective = self.rec_lambda * self.loss_reconstruct + \
                self.wae_lambda * self.penalty + \
                stay_lambda * self.stay_loss

        # Extra costs if any
        if 'w_aef' in opts and opts['w_aef'] > 0:
            improved_wae.add_aefixedpoint_cost(opts, self)

        self.blurriness = self.compute_blurriness()

        if opts['e_pretrain']:
            self.loss_pretrain = self.pretrain_loss()
        else:
            self.loss_pretrain = None

        self.add_least_gaussian2d_ops()

        # -- Optimizers, savers, etc

        self.add_optimizers()
        self.add_savers()
        self.init = tf.global_variables_initializer()

    def add_to_log(self, key, value):
        self.tensors_to_log[key] = value

    def get_tensors_to_log(self):
        keys = ['sinkhorn_ot', 'mmd_linear', 'mmd']  
        for key in keys:
            if key not in self.tensors_to_log:
                self.tensors_to_log[key] = tf.constant(0.0)
        return self.tensors_to_log

    def add_nat_placeholders(self):
        opts = self.opts
        # TODO implement options: spherical, etc..
        self.nat_targets_np = self.sample_pz(self.opts['nat_size'])
        #self.nat_targets = tf.placeholder(tf.float32, shape=(opts['nat_size'], opts['zdim']))
        self.nat_targets = tf.Variable(self.nat_targets_np, dtype=tf.float32, trainable=False)
        self.x_latents = tf.Variable(tf.zeros((opts['nat_size'], opts['zdim'])), dtype=tf.float32, trainable=False)
        self.batch_indices_mod = tf.placeholder(tf.int64, shape=(opts['batch_size'],))
        self.nat_sparse_indices = tf.placeholder(tf.int64, shape=(None, 2)) #opts['nat_sparse_indices_num']
        self.x_rec_losses_np = 100000.0*np.ones((self.train_size,)) #tf.Variable(100000.0*tf.ones((self.train_size, )), dtype=tf.float32, trainable=False)

    def add_nat_tensors(self):
        opts = self.opts
        n = opts['nat_size']
        bs = opts['batch_size']
        cut = int(bs*opts['mover_ratio'])

        self.movers  = self.batch_indices_mod[:cut]
        self.stayers = self.batch_indices_mod[cut:]

        x_latents_with_current_batch = tf.stop_gradient(tf.boolean_mask(self.x_latents,
            tf.sparse_to_dense(
                sparse_indices=self.movers,
                default_value=1.0,
                sparse_values=0.0,
                output_shape=[n], validate_indices=False
                )
            ))
        x_latents_with_current_batch = tf.concat([x_latents_with_current_batch, 
            self.encoded[:cut]], axis=0)
        x_latents_with_current_batch = tf.reshape(x_latents_with_current_batch, shape=(n, opts['zdim']))
        self.x_latents_with_current_batch = x_latents_with_current_batch

        self.nat_targets_update_ph = tf.placeholder(self.nat_targets.dtype, shape=self.nat_targets.get_shape())
        self.update_nat_targets_op = self.nat_targets.assign(self.nat_targets_update_ph)

        self.x_latents_update_ph = tf.placeholder(self.x_latents.dtype, shape=self.x_latents.get_shape())
        self.update_x_latents_op = self.x_latents.assign(self.x_latents_update_ph)

        self.su_ids_to_update_ph = tf.placeholder(tf.int64, shape=(self.opts['recalculate_size'],))
        self.su_latents_ph = tf.placeholder(tf.float32, shape=(self.opts['recalculate_size'], self.opts['zdim']))
        self.scatter_update_x_latents_op = tf.scatter_update(self.x_latents, self.su_ids_to_update_ph, self.su_latents_ph)

    def resample_nat_targets(self):
        self.nat_targets_np = self.sample_pz(self.opts['nat_size'])
        self.sess.run(self.update_nat_targets_op, feed_dict={self.nat_targets_update_ph: self.nat_targets_np})

    def add_inputs_placeholders(self):
        opts = self.opts
        shape = self.data_shape
        data = tf.placeholder(
            tf.float32, [None] + shape, name='real_points_ph')
        noise = tf.placeholder(
            tf.float32, [None] + [opts['zdim']], name='noise_ph')
        self.sample_points = data
        self.sample_noise = noise

    def add_training_placeholders(self):
        opts = self.opts
        decay = tf.placeholder(tf.float32, name='rate_decay_ph')
        wae_lambda = tf.placeholder(tf.float32, name='lambda_ph')
        is_training = tf.placeholder(tf.bool, name='is_training_ph')
        rec_lambda = tf.placeholder(tf.float32, name='rec_lambda_ph')
        zxz_lambda = tf.placeholder(tf.float32, name='zxz_lambda_ph')

        self.lr_decay = decay
        self.wae_lambda = wae_lambda
        self.is_training = is_training
        self.rec_lambda = rec_lambda
        self.zxz_lambda = zxz_lambda

        #self.ids_to_update_ph = tf.placeholder(tf.int32, shape=(self.opts['batch_size'])
        #self.values_for_update_ph = tf.placeholder(tf.float32, shape=(self.opts['batch_size'])
        #self.update_latents_op = tf.scatter_update(self.x_latents, self.ids_to_update_ph, latents)

    def stay_loss(self):

        opts = self.opts
        n = opts['nat_size']
        bs = opts['batch_size']
        cut = int(bs*opts['mover_ratio'])

        old_positions = tf.stop_gradient(tf.boolean_mask(self.x_latents,
            tf.sparse_to_dense(
                sparse_indices=self.stayers,
                default_value=0.0,
                sparse_values=1.0,
                output_shape=[n], validate_indices=False
                )
            ))

        new_positions = self.encoded[cut:]
        stay_loss = tf.reduce_sum(tf.square(new_positions - old_positions))
       # stay_loss = tf.abs(tf.reduce_sum(new_positions-old_positions))

        return stay_loss

    # TODO takes sparse indices from self.nat_sparse_indices, not good for test data.
    def sinkhorn_loss(self, sample_qz, sample_pz):
        opts = self.opts

        #global_step = tf.train.get_or_create_global_step()
        #decayed_epsilon = tf.train.cosine_decay_restarts(learning_rate=args.epsilon, global_step=global_step, first_decay_steps=20, alpha=0.0001)
        decayed_epsilon = tf.constant(opts['sinkhorn_epsilon'])

        if opts['sinkhorn_sparse']:
            OT, P_temp, P, f, g, C = sinkhorn.SparseSinkhornLoss(sample_qz, sample_pz, sparse_indices=self.nat_sparse_indices, epsilon=decayed_epsilon, niter=opts['sinkhorn_iters'])
        else:
            OT, P_temp, P, f, g, C = sinkhorn.SinkhornLoss(sample_qz, sample_pz, epsilon=decayed_epsilon, niter=opts['sinkhorn_iters'])

        return OT

    def zxz_loss(self):
        opts = self.opts

        z = self.sample_noise
        decoded, decoded_logits = decoder(opts, reuse=True, noise=z, is_training=self.is_training)
        zxz, _ = encoder(opts, reuse=True, inputs=decoded, is_training=self.is_training)
        loss = tf.reduce_sum(tf.square(zxz - z), axis=[1])
        self.zxz_loss = tf.reduce_mean(loss)
        return self.zxz_loss

    def pretrain_loss(self):
        opts = self.opts
        # Adding ops to pretrain the encoder so that mean and covariance
        # of Qz will try to match those of Pz
        mean_pz = tf.reduce_mean(self.sample_noise, axis=0, keep_dims=True)
        mean_qz = tf.reduce_mean(self.encoded, axis=0, keep_dims=True)
        mean_loss = tf.reduce_mean(tf.square(mean_pz - mean_qz))
        cov_pz = tf.matmul(self.sample_noise - mean_pz,
                           self.sample_noise - mean_pz, transpose_a=True)
        cov_pz /= opts['e_pretrain_sample_size'] - 1.
        cov_qz = tf.matmul(self.encoded - mean_qz,
                           self.encoded - mean_qz, transpose_a=True)
        cov_qz /= opts['e_pretrain_sample_size'] - 1.
        cov_loss = tf.reduce_mean(tf.square(cov_pz - cov_qz))
        return mean_loss + cov_loss

    def add_savers(self):
        opts = self.opts
        saver = tf.train.Saver(max_to_keep=10)
        tf.add_to_collection('real_points_ph', self.sample_points)
        tf.add_to_collection('noise_ph', self.sample_noise)
        tf.add_to_collection('is_training_ph', self.is_training)
        if self.enc_mean is not None:
            tf.add_to_collection('encoder_mean', self.enc_mean)
            tf.add_to_collection('encoder_var', self.enc_sigmas)
        if opts['e_noise'] == 'implicit':
            tf.add_to_collection('encoder_A', self.encoder_A)
        tf.add_to_collection('encoder', self.encoded)
        tf.add_to_collection('decoder', self.decoded)
        if self.loss_gan:
            tf.add_to_collection('disc_logits_Pz', self.loss_gan[1])
            tf.add_to_collection('disc_logits_Qz', self.loss_gan[2])
        self.saver = saver

    def add_least_gaussian2d_ops(self):
        """ Add ops searching for the 2d plane in z_dim hidden space
            corresponding to the 'least Gaussian' look of the sample
        """

        opts = self.opts

        with tf.variable_scope('leastGaussian2d'):
            # Projection matrix which we are going to tune
            sample = tf.placeholder(
                tf.float32, [None, opts['zdim']], name='sample_ph')
            v = tf.get_variable(
                "proj_v", [opts['zdim'], 1],
                tf.float32, tf.random_normal_initializer(stddev=1.))
            u = tf.get_variable(
                "proj_u", [opts['zdim'], 1],
                tf.float32, tf.random_normal_initializer(stddev=1.))
            npoints = tf.cast(tf.shape(sample)[0], tf.int32)

            # First we need to make sure projection matrix is orthogonal

            v_norm = tf.nn.l2_normalize(v, 0)
            dotprod = tf.reduce_sum(tf.multiply(u, v_norm))
            u_ort = u - dotprod * v_norm
            u_norm = tf.nn.l2_normalize(u_ort, 0)
            Mproj = tf.concat([v_norm, u_norm], 1)
            sample_proj = tf.matmul(sample, Mproj)
            a = tf.eye(npoints)
            a -= tf.ones([npoints, npoints]) / tf.cast(npoints, tf.float32)
            b = tf.matmul(sample_proj, tf.matmul(a, a), transpose_a=True)
            b = tf.matmul(b, sample_proj)
            # Sample covariance matrix
            covhat = b / (tf.cast(npoints, tf.float32) - 1)
            gcov = opts['pz_scale'] ** 2.  * tf.eye(2)
            # l2 distance between sample cov and the Gaussian cov
            projloss =  tf.reduce_sum(tf.square(covhat - gcov))
            # Also account for the first moment, i.e. expected value
            projloss += tf.reduce_sum(tf.square(tf.reduce_mean(sample_proj, 0)))
            # We are maximizing
            projloss = -projloss
            optim = tf.train.AdamOptimizer(0.001, 0.9)
            optim = optim.minimize(projloss, var_list=[v, u])

        self.proj_u = u_norm
        self.proj_v = v_norm
        self.proj_sample = sample
        self.proj_covhat = covhat
        self.proj_loss = projloss
        self.proj_opt = optim

    def matching_penalty(self):
        opts = self.opts
        loss_gan = None
        if opts['z_test_scope'] == 'global':
            sample_qz = self.x_latents_with_current_batch
            sample_pz = self.nat_targets
        else:
            sample_qz = self.encoded
            sample_pz = self.sample_noise

        if opts['z_test'] == 'gan':
            loss_gan, loss_match = self.gan_penalty(sample_qz, sample_pz)
        elif opts['z_test'] == 'mmd':
            loss_match = self.mmd_penalty(sample_qz, sample_pz)
        elif opts['z_test'] == 'mmd_linear':
            mmd = self.mmd_linear(sample_qz, sample_pz)
        elif opts['z_test'] == 'mmdpp':
            loss_match = improved_wae.mmdpp_penalty(
                opts, self, sample_pz)
        elif opts['z_test'] == 'mmdppp':
            loss_match = improved_wae.mmdpp_1d_penalty(
                opts, self, sample_pz)
        elif opts['z_test'] == 'sinkhorn':
            loss_match = self.sinkhorn_loss(sample_qz, sample_pz)
            self.add_to_log("sinkhorn_ot", loss_match)
        elif opts['z_test'] == 'sliced_wae':
            loss_match = self.sliced_wae_loss(sample_qz, sample_pz)
        elif opts['z_test'] == 'sliced_wae_adv':
            loss_match = self.sliced_wae_adversarial_loss(sample_qz, sample_pz)
       # elif opts['z_test'] == 'sinkhorn_stay_loss':
       #     loss_match, stay_loss = self.sinkhorn_loss_with_stay(sample_qz, sample_pz)
        else:
            assert False, 'Unknown penalty %s' % opts['z_test']
        return loss_match, loss_gan

    def sliced_wae_loss(self, sample_qz, sample_pz):
        opts = self.opts
        L = 250 #number of projections
        endim = opts['zdim']

        theta=np.asarray([w/np.sqrt((w**2).sum()) for w in np.random.normal(size=(L,endim))])
        k = utils.get_batch_size(sample_qz)
        k = tf.cast(k, tf.int32)
        theta = tf.Variable(theta)
        theta = tf.cast(theta, tf.float32)

        proj_pz = tf.matmul(sample_pz, tf.transpose(theta))
        proj_qz = tf.matmul(sample_qz, tf.transpose(theta))

        W2=(tf.nn.top_k(tf.transpose(proj_pz),k).values
           - tf.nn.top_k(tf.transpose(proj_qz),k).values)**2

        W2 = tf.math.reduce_sum(W2)

        return W2


    def sliced_wae_adversarial_loss(self, sample_qz, sample_pz):
        opts = self.opts
        endim = opts['zdim']

        s1, u1, theta1 = tf.linalg.svd(sample_qz)
      #  s2, u2, theta2 = tf.linalg.svd(sample.pz)
        k = utils.get_batch_size(sample_qz)
        k = tf.cast(k, tf.int32)
        theta1 = tf.slice(tf.cast(theta1, tf.float32), [0,0], [1,2])
      # theta2 = tf.cast(theta2, tf.float32)

        proj_pz1 = tf.matmul(sample_pz, tf.transpose(theta1))
        proj_qz1 = tf.matmul(sample_qz, tf.transpose(theta1))

      # proj_pz2 = tf.matmul(sample_pz, tf.transpose(theta2))
      # proj_qz2 = tf.matmul(sample_qz, tf.transpose(theta2))

        W2 = (tf.nn.top_k(tf.transpose(proj_pz1),k).values
            - tf.nn.top_k(tf.transpose(proj_qz1),k).values)**2
        W2 = tf.math.reduce_sum(W2)

        return W2


    def mmd_linear(self, sample_qz, sample_pz):
        opts = self.opts
        sigma2_p = opts['pz_scale'] ** 2
        kernel = opts['mmd_kernel']
        n = utils.get_batch_size(sample_qz)
        n = tf.cast(n, tf.int32)
        nf = tf.cast(n, tf.float32)

        with tf.Session() as sess:
   	     half = n.eval()//2
        even_indices = [2*i for i in range(half+1)]
        odd_indices = [2*i+1 for i in range(half+1)]
        pz_even = tf.gather(sample_pz, even_indices)
        pz_odd = tf.gather(sample_pz, odd_indices)
        qz_even = tf.gather(sample_qz, even_indices)
        qz_odd = tf.gather(sample_qz, odd_indices)


        norms_even_pz = tf.reduce_sum(tf.square(pz_even), axis=1)
        norms_odd_pz = tf.reduce_sum(tf.square(pz_odd), axis=1)
        neighbor_dotprods_pz = tf.reduce_sum(tf.math.multiply(pz_even,pz_odd), axis=1)
        distances_pz = norms_even_pz + norms_odd_pz -2. * neighbor_dotprods_pz

        norms_even_qz = tf.reduce_sum(tf.square(qz_even), axis=1)
        norms_odd_qz = tf.reduce_sum(tf.square(qz_odd), axis=1)
        neighbor_dotprods_qz = tf.reduce_sum(tf.math.multiply(qz_even,qz_odd), axis=1)
        distances_qz = norms_even_qz + norms_odd_qz -2. * neighbor_dotprods_qz

        cross_dotprods_1 = tf.reduce_sum(tf.math.multiply(pz_even,qz_odd), axis=1)
        cross_dotprods_2 = tf.reduce_sum(tf.math.multiply(pz_odd,qz_even), axis=1)
        cross_distances_1 = norms_even_pz + norms_odd_qz - 2. * cross_dotprods_1
        cross_distances_2 = norms_odd_pz + norms_even_qz - 2. * cross_dotprods_2

        if kernel == 'RBF':
            sigma2_k = tf.nn.top_k(cross_distances_1, half).values[half - 1]
            sigma2_k += tf.nn.top_k(distances_qz, half).values[half - 1]

            res1 = tf.exp( - distances_qz / 2. / sigma2_k)
            res1 += tf.exp( - distances_pz / 2. / sigma2_k)
            res1 = tf.reduce_sum(res1) / half
            res2 = tf.exp( - cross_distances_1 / 2. / sigma2_k)
            res2 += tf.exp( - cross_distances_2 / 2. / sigma2_k)
            res2 = tf.reduce_sum(res2)  / half
            stat = res1 - res2

        elif kernel == 'IMQ':
            # k(x, y) = C / (C + ||x - y||^2)
            # C = tf.nn.top_k(tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            # C += tf.nn.top_k(tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
            if opts['pz'] == 'normal':
                Cbase = 2. * opts['zdim'] * sigma2_p
            elif opts['pz'] == 'sphere':
                Cbase = 2.
            elif opts['pz'] == 'uniform':
                # E ||x - y||^2 = E[sum (xi - yi)^2]
                #               = zdim E[(xi - yi)^2]
                #               = const * zdim
                Cbase = opts['zdim']
            stat = 0.
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                C = Cbase * scale
                res1 = C / (C + distances_qz)
                res1 += C / (C + distances_pz)
                res1 = tf.reduce_sum(res1) / half
                res2 = C / (C + cross_distances_1)
                res2 += C / (C + cross_distances_2)
                res2 = tf.reduce_sum(res2) / half
                stat += res1 - res2

        self.add_to_log("mmd_linear", stat)

        return stat


    def mmd_penalty(self, sample_qz, sample_pz):
        opts = self.opts
        sigma2_p = opts['pz_scale'] ** 2
        kernel = opts['mmd_kernel']
        n = utils.get_batch_size(sample_qz)
        n = tf.cast(n, tf.int32)
        nf = tf.cast(n, tf.float32)
        half_size = (n * n - n) // 2

        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keep_dims=True)
        dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
        distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keep_dims=True)
        dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
        distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

        dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

        # if opts['verbose']:
        #     distances = tf.Print(
        #         distances,
        #         [tf.nn.top_k(tf.reshape(distances_qz, [-1]), 1).values[0]],
        #         'Maximal Qz squared pairwise distance:')
        #     distances = tf.Print(distances, [tf.reduce_mean(distances_qz)],
        #                         'Average Qz squared pairwise distance:')

        #     distances = tf.Print(
        #         distances,
        #         [tf.nn.top_k(tf.reshape(distances_pz, [-1]), 1).values[0]],
        #         'Maximal Pz squared pairwise distance:')
        #     distances = tf.Print(distances, [tf.reduce_mean(distances_pz)],
        #                         'Average Pz squared pairwise distance:')

        if kernel == 'RBF':
            # Median heuristic for the sigma^2 of Gaussian kernel
            sigma2_k = tf.nn.top_k(
                tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            sigma2_k += tf.nn.top_k(
                tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
            # Maximal heuristic for the sigma^2 of Gaussian kernel
            # sigma2_k = tf.nn.top_k(tf.reshape(distances_qz, [-1]), 1).values[0]
            # sigma2_k += tf.nn.top_k(tf.reshape(distances, [-1]), 1).values[0]
            # sigma2_k = opts['latent_space_dim'] * sigma2_p
            if opts['verbose']:
                sigma2_k = tf.Print(sigma2_k, [sigma2_k], 'Kernel width:')
            res1 = tf.exp( - distances_qz / 2. / sigma2_k)
            res1 += tf.exp( - distances_pz / 2. / sigma2_k)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = tf.exp( - distances / 2. / sigma2_k)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat = res1 - res2
        elif kernel == 'IMQ':
            # k(x, y) = C / (C + ||x - y||^2)
            # C = tf.nn.top_k(tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            # C += tf.nn.top_k(tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
            if opts['pz'] == 'normal':
                Cbase = 2. * opts['zdim'] * sigma2_p
            elif opts['pz'] == 'sphere':
                Cbase = 2.
            elif opts['pz'] == 'uniform':
                # E ||x - y||^2 = E[sum (xi - yi)^2]
                #               = zdim E[(xi - yi)^2]
                #               = const * zdim
                Cbase = opts['zdim']
            stat = 0.

          # with tf.device('/gpu:1'):
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                C = Cbase * scale
                res1 = C / (C + distances_qz)
                res1 += C / (C + distances_pz)
                res1 = tf.multiply(res1, 1. - tf.eye(n))
                res1 = tf.reduce_sum(res1) / (nf * nf - nf)
                res2 = C / (C + distances)
                res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
                stat += res1 - res2

        self.add_to_log("mmd", stat)

        return stat

    def gan_penalty(self, sample_qz, sample_pz):
        opts = self.opts
        # Pz = Qz test based on GAN in the Z space
        logits_Pz = z_adversary(opts, sample_pz)
        logits_Qz = z_adversary(opts, sample_qz, reuse=True)
        loss_Pz = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_Pz, labels=tf.ones_like(logits_Pz)))
        loss_Qz = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_Qz, labels=tf.zeros_like(logits_Qz)))
        loss_Qz_trick = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_Qz, labels=tf.ones_like(logits_Qz)))
        loss_adversary = self.wae_lambda * (loss_Pz + loss_Qz)
        # Non-saturating loss trick
        loss_match = loss_Qz_trick
        return (loss_adversary, logits_Pz, logits_Qz), loss_match

    @staticmethod
    def reconstruction_loss(opts, real, reconstr):
        # real = self.sample_points
        # reconstr = self.reconstructed
        if opts['cost'] == 'l2':
            # c(x,y) = ||x - y||_2
            loss = tf.reduce_sum(tf.square(real - reconstr), axis=[1, 2, 3])
            per_sample_loss = tf.sqrt(1e-08 + loss)
            loss = 0.2 * tf.reduce_mean(tf.sqrt(1e-08 + loss))
        elif opts['cost'] == 'l2sq':
            # c(x,y) = ||x - y||_2^2
            loss = tf.reduce_sum(tf.square(real - reconstr), axis=[1, 2, 3])
            per_sample_loss = loss
            loss = 0.05 * tf.reduce_mean(loss)
        elif opts['cost'] == 'l1':
            # c(x,y) = ||x - y||_1
            loss = tf.reduce_sum(tf.abs(real - reconstr), axis=[1, 2, 3])
            per_sample_loss = loss
            loss = 0.02 * tf.reduce_mean(loss)
        else:
            assert False, 'Unknown cost function %s' % opts['cost']
        return loss, per_sample_loss

    def compute_blurriness(self):
        images = self.sample_points
        sample_size = tf.shape(self.sample_points)[0]
        # First convert to greyscale
        if self.data_shape[-1] > 1:
            # We have RGB
            images = tf.image.rgb_to_grayscale(images)
        # Next convolve with the Laplace filter
        lap_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        lap_filter = lap_filter.reshape([3, 3, 1, 1])
        conv = tf.nn.conv2d(images, lap_filter,
                            strides=[1, 1, 1, 1], padding='VALID')
        _, lapvar = tf.nn.moments(conv, axes=[1, 2, 3])
        return lapvar


    def optimizer(self, lr, decay=1.):
        opts = self.opts
        lr *= decay
        if opts["optimizer"] == "sgd":
            return tf.train.GradientDescentOptimizer(lr)
        elif opts["optimizer"] == "adam":
            return tf.train.AdamOptimizer(lr, beta1=opts["adam_beta1"])
        else:
            assert False, 'Unknown optimizer.'

    def add_optimizers(self):
        opts = self.opts
        lr = opts['lr']
        lr_adv = opts['lr_adv']
        z_adv_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='z_adversary')
        encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        ae_vars = encoder_vars + decoder_vars

        if opts['verbose']:
            logging.error('Param num in AE: %d' % \
                    np.sum([np.prod([int(d) for d in v.get_shape()]) \
                    for v in ae_vars]))

        # Auto-encoder optimizer
        opt = self.optimizer(lr, self.lr_decay)
        self.ae_opt = opt.minimize(loss=self.wae_objective,
                              var_list=encoder_vars + decoder_vars)


        opt = self.optimizer(lr, self.lr_decay)
        self.zxz_opt = opt.minimize(loss=self.zxz_lambda*self.zxz_loss, var_list=decoder_vars)

        """
        self.ot_grads_and_vars = opt.compute_gradients(loss=self.ot_loss, var_list=self.x_latents)
        self.ae_grads_and_vars = opt.compute_gradients(loss=self.wae_objective, var_list=encoder_vars + decoder_vars)

        self.merged_grads_and_vars = []
        for g, v in self.ae_grads_and_vars:
            if v == self.encoded:
                g += self.ae_grads_and_vars[0][0]
                self.merged_grads_and_vars.append((g, v))
            else:
                self.merged_grads_and_vars.append((g, v))

        #self.ot_apply_grads = opt.apply_gradients(self.ot_grads_and_vars)
        self.ae_apply_grads = opt.apply_gradients(self.ae_grads_and_vars)
        self.merged_apply_grads = opt.apply_gradients(self.merged_grads_and_vars)
        """

        # Discriminator optimizer for WAE-GAN
        if opts['z_test'] == 'gan':
            opt = self.optimizer(lr_adv, self.lr_decay)
            self.z_adv_opt = opt.minimize(
                loss=self.loss_gan[0], var_list=z_adv_vars)
        else:
            self.z_adv_opt = None

        # Encoder optimizer
        if opts['e_pretrain']:
            opt = self.optimizer(lr)
            self.pretrain_opt = opt.minimize(loss=self.loss_pretrain,
                                             var_list=encoder_vars)
        else:
            self.pretrain_opt = None


    def sample_pz(self, num=100):
        opts = self.opts
        noise = None
        distr = opts['pz']
        if distr == 'uniform':
            noise = np.random.uniform(
                -1, 1, [num, opts["zdim"]]).astype(np.float32)
        elif distr in ('normal', 'sphere'):
            mean = np.zeros(opts["zdim"])
            cov = np.identity(opts["zdim"])
            noise = np.random.multivariate_normal(
                mean, cov, num).astype(np.float32)
            if distr == 'sphere':
                noise = noise / np.sqrt(
                    np.sum(noise * noise, axis=1))[:, np.newaxis]
        return opts['pz_scale'] * noise

    def pretrain_encoder(self, data):
        opts = self.opts
        steps_max = 200
        batch_size = opts['e_pretrain_sample_size']
        for step in range(steps_max):
            data_ids = np.random.choice(self.train_size, min(self.train_size, batch_size),
                                        replace=False)
            batch_images = data.data[data_ids].astype(np.float)
            batch_noise =  self.sample_pz(batch_size)

            [_, loss_pretrain] = self.sess.run(
                [self.pretrain_opt,
                 self.loss_pretrain],
                feed_dict={self.sample_points: batch_images,
                           self.sample_noise: batch_noise,
                           self.is_training: True})

            if opts['verbose']:
                logging.error('Step %d/%d, loss=%f' % (
                    step, steps_max, loss_pretrain))

            if loss_pretrain < 0.1:
                break

    def least_gaussian_2d(self, X):
        """
        Given a sample X of shape (n_points, n_z) find 2d plain
        such that projection looks least Gaussian
        """
        opts = self.opts
        with self.sess.as_default(), self.sess.graph.as_default():
            sample = self.proj_sample
            optim = self.proj_opt
            loss = self.proj_loss
            u = self.proj_u
            v = self.proj_v

            covhat = self.proj_covhat
            proj_mat = tf.concat([v, u], 1).eval()
            dot_prod = -1
            best_of_runs = 10e5 # Any positive value would do
            updated = False
            for _ in range(3):
                # We will run 3 times from random inits
                loss_prev = 10e5 # Any positive value would do
                proj_vars = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope='leastGaussian2d')
                self.sess.run(tf.variables_initializer(proj_vars))
                step = 0
                for _ in range(5000):
                    self.sess.run(optim, feed_dict={sample:X})
                    step += 1
                    if step % 10 == 0:
                        loss_cur = loss.eval(feed_dict={sample: X})
                        rel_imp = abs(loss_cur - loss_prev) / abs(loss_prev)
                        if rel_imp < 1e-2:
                            break
                        loss_prev = loss_cur
                loss_final = loss.eval(feed_dict={sample: X})
                if loss_final < best_of_runs:
                    updated = True
                    best_of_runs = loss_final
                    proj_mat = tf.concat([v, u], 1).eval()
                    dot_prod = tf.reduce_sum(tf.multiply(u, v)).eval()
        if not updated:
            logging.error('WARNING: possible bug in the worst 2d projection')
        return proj_mat, dot_prod


    def encode(self, x, batch_size):
        latents_list = []
        for k in range(x.shape[0] // batch_size):
            batch_images_temp = x[k*batch_size:(k+1)*batch_size]
            batch_latents_temp, = self.sess.run([self.encoded], feed_dict={self.is_training: False, self.sample_points: batch_images_temp})
            latents_list.append(batch_latents_temp)
        latents = np.concatenate(latents_list, axis=0)
        return latents


    def recalculate_x_latents(self, data, train_size, batch_size, overwrite_placeholder=True, ids=None):
        # Calculate latent image of the full dataset
        if ids is not None:
            data_temp = data.data[ids]
        else:
            data_temp = data.data[:self.opts['nat_size']]
 
        latents = self.encode(data_temp, batch_size)
        print("Recalculated latents, shape: ", latents.shape)

        if overwrite_placeholder:
            if ids is not None:
                ids_to_update = [i for i in range(self.nat_pos, self.nat_pos + self.opts['recalculate_size'])]
                # Old eval op: tf.scatter_update(self.x_latents, ids_to_update, latents).eval(session=self.sess)
                self.sess.run(self.scatter_update_x_latents_op, feed_dict={self.su_ids_to_update_ph: ids_to_update, self.su_latents_ph: latents})
                self.x_latents_np[ids_to_update] = latents
                self.nat_pos = (self.nat_pos + self.opts['recalculate_size']) % self.opts['nat_size']
            else:
                # Old eval op: self.x_latents.assign(latents).eval(session=self.sess)
                self.sess.run(self.update_x_latents_op, feed_dict={self.x_latents_update_ph: latents})
                self.x_latents_np = latents
                self.nat_pos = 0
        return latents


    def sparsifier_factory(self, sources, targets):
        use_sparse = self.opts['sinkhorn_sparse']
        sparsifier_kind = self.opts['sinkhorn_sparsifier']
        k = 10
        batch_size = 200
        sess = self.sess
        n = sources.get_shape().as_list()[0]
        m = targets.get_shape().as_list()[0]
        if not use_sparse or sparsifier_kind == "dense":
            sparsifier = None
        elif sparsifier_kind == "topk":
            sparsifier = sparsifiers.TfTopkSparsifier(sources, targets, k, sess, batch_size=batch_size)
        elif sparsifier_kind == "twoway-topk":
            sparsifier = sparsifiers.TfTwoWayTopkSparsifier(sources, targets, k, sess, batch_size=batch_size)
        elif sparsifier_kind == "random":
            sparsifier = sparsifiers.RandomSparsifier(n, m, k * n, resample=True)
        elif sparsifier_kind == "random-without-resample":
            sparsifier = sparsifiers.RandomSparsifier(n, m, k * n, resample=False)
        elif sparsifier_kind == "mishmash":
            twoway_sparsifier = sparsifiers.TfTwoWayTopkSparsifier(sources, targets, 0, sess, batch_size=batch_size)
            random_sparsifier = sparsifiers.RandomSparsifier(n, m, k * n // 2, resample=True)
            sparsifier = sparsifiers.SparsifierCombinator(twoway_sparsifier, random_sparsifier)
        else:
            assert False, "unknown sparsifier kind"

        return sparsifier


    def train(self, data):
        opts = self.opts
        if opts['verbose']:
            logging.error(opts)
        logging.error('Training WAE')
        losses = []
        losses_rec = []
        losses_match = []
        blurr_vals = []
        encoding_changes = []
        enc_test_prev = None
        batches_num = self.train_size // opts['batch_size']
        self.num_pics = opts['plot_num_pics']
        self.fixed_noise = self.sample_pz(opts['plot_num_pics'])

        self.sess.run(self.init)

        if opts['e_pretrain']:
            logging.error('Pretraining the encoder')
            self.pretrain_encoder(data)
            logging.error('Pretraining the encoder done.')


        self.start_time = time.time()
        counter = 0
        decay = 1.
        wae_lambda = opts['lambda']
        rec_lambda = opts['rec_lambda']
        zxz_lambda = opts['zxz_lambda']
        batch_size = opts['batch_size']


        # Weights of the extra costs
        extra_cost_weights = []
        if 'w_aef' in opts and opts['w_aef'] > 0:
            extra_cost_weights.append((self.w_aefixedpoint, opts['w_aef']))
        # Variables for dynamic updates
        wait = 0
        wait_lambda = 0

        real_blurr = self.sess.run(
            self.blurriness,
            feed_dict={self.sample_points: data.data[:self.num_pics]})
        logging.error('Real pictures sharpness = %.5f' % np.min(real_blurr))


        gradlog_file = open("gradlog.txt", "w")
        
        VIDEO_SIZE = 512
        with FFMPEG_VideoWriter(opts['name'] + 'out.mp4', (VIDEO_SIZE, VIDEO_SIZE), 3.0) as video:
          #if True:

          self.recalculate_x_latents(data, self.train_size, batch_size, overwrite_placeholder=True, ids=None)

          self.sparsifier = self.sparsifier_factory(self.x_latents, self.nat_targets)
          
          for epoch in range(opts["epoch_num"]):

            # Update learning rate if necessary

            if opts['lr_schedule'] == "manual":
                if epoch == 30:
                    decay = decay / 2.
                if epoch == 50:
                    decay = decay / 5.
                if epoch == 100:
                    decay = decay / 10.
            elif opts['lr_schedule'] == 'manual_proportional':
                enum = opts['epoch_num']
                if epoch == 3* enum//10:
                    decay = decay / 2.
                if epoch == enum//2:
                    decay = decay / 5.
            elif opts['lr_schedule'] == "manual_smooth":
                enum = opts['epoch_num']
                decay_t = np.exp(np.log(100.) / enum)
                decay = decay / decay_t

            elif opts['lr_schedule'] != "plateau":
                assert type(opts['lr_schedule']) == float
                decay = 1.0 * 10**(-epoch / float(opts['lr_schedule']))

            # Save the model

            if epoch > 0 and epoch % opts['save_every_epoch'] == 0:
                self.saver.save(self.sess,
                                 os.path.join(opts['work_dir'],
                                              'checkpoints',
                                              'trained-wae'),
                                 global_step=counter)

            # Iterate over batches

            if self.opts['nat_resampling'] == 'epoch':
                self.resample_nat_targets()

            #self.recalculate_x_latents(data, self.train_size, batch_size, overwrite_placeholder=True, ids=None)
            if self.sparsifier is not None:
                self.sparsifier.on_epoch_begin()

            for it in range(batches_num):

                if self.opts['nat_resampling'] == 'batch':
                    self.resample_nat_targets()

                # Sample batches of data points and Pz noise
                if (self.opts['feed_by_score_from_epoch'] != -1) and (self.opts['feed_by_score_from_epoch'] <= epoch+1):
                    data_ids = np.argpartition(self.x_rec_losses_np, -opts['batch_size'])[-opts['batch_size']:]
                    all_data_ids = np.random.choice(self.train_size, opts['recalculate_size'], replace=False)
                elif self.opts['shuffle']:
                    assert opts['recalculate_size']>=opts['batch_size'], "recalculate_size must be as large as batch_size"
                    all_data_ids = np.random.choice(self.train_size, opts['recalculate_size'], replace=False)
                    data_ids = all_data_ids[:opts['batch_size']]
                else:
                    rnd_it = random.randint(0, batches_num-1)
                    #data_ids = np.arange(rnd_it*opts['batch_size'], (rnd_it+1)*opts['batch_size'])
                    #all_data_ids = data_ids
                    #all_data_ids = np.arange((rnd_it*opts['recalculate_size']) % self.train_size, ((rnd_it+1)*opts['recalculate_size']) % self.train_size)
                    all_data_ids = np.arange(0, self.train_size)
                    data_ids = np.random.choice(self.train_size, opts['batch_size'], replace=False)


                #data_ids_mod = np.array([i for i in range(self.nat_pos, self.nat_pos + self.opts['batch_size'])])
                data_ids_mod = data_ids
                batch_images = data.data[data_ids].astype(np.float)
                batch_noise = self.sample_pz(opts['batch_size'])

                if counter % opts['print_every'] == 0:
                    (x_latents_np, nat_targets_np) = self.sess.run([self.x_latents, self.nat_targets], feed_dict={self.sample_points: batch_images, self.is_training:False})
                    print("frame,", nat_targets_np.shape)
                    x_latents_unif = x_latents_np[:, :2]
                    nat_targets_unif = nat_targets_np[:, :2]
                    if opts['exp'] == 'celebA':
                        x_latents_unif = norm.cdf(x_latents_unif) * 2 - 1
                        nat_targets_unif = norm.cdf(nat_targets_unif) * 2 - 1                            
                    frame = sinkhorn.draw_edges(x_latents_unif, nat_targets_unif, VIDEO_SIZE, radius=1.5, edges=False)
                    video.write_frame(frame)
                    print("frame")
                    import synth_data
                    covered = synth_data.covered_area(x_latents_unif, resolution=400, radius=5)
                    print("covered_area", x_latents_unif.shape, covered)
                    if 'NEPTUNE_API_TOKEN' in os.environ:
                        neptune.send_metric('covered_area', x=counter, y=covered)

                # Update encoder and decoder
                feed_d = {
                    self.sample_points: batch_images,
                    self.sample_noise: batch_noise,
                    self.lr_decay: decay,
                    self.wae_lambda: wae_lambda,
                    self.rec_lambda: rec_lambda,
                    self.is_training: True,
                    self.batch_indices_mod: data_ids_mod}

                if self.sparsifier is not None:
                    if (self.opts['sparsifier_freq'] is None) or (it % self.opts['sparsifier_freq'] == 0):
                        self.nat_sparse_indices_np = self.sparsifier.indices()
                    feed_d[self.nat_sparse_indices] = self.nat_sparse_indices_np

                #ot_grads_and_vars_np = self.sess.run([self.ot_grads_and_vars], feed_dict=feed_d)

                for (ph, val) in extra_cost_weights:
                    feed_d[ph] = val

                run_ops = [
                    self.ae_opt,
                    self.wae_objective,
                    self.loss_reconstruct,
                    self.penalty,
                    self.stay_loss,
                    self.per_sample_rec_loss,
                    #self.rec_grad, self.latent_grad
                ]
                len_orig_run_ops = len(run_ops)
                for key, value in self.get_tensors_to_log().items():
                    run_ops.append(value)

                run_result = self.sess.run(run_ops, feed_dict=feed_d)
                [_, loss, loss_rec, loss_match, stay_loss, per_sample_rec_loss_np] = run_result[:len_orig_run_ops]

                run_result_dict = {}
                i = 0
                for key, value in self.get_tensors_to_log().items():
                    run_result_dict[key] = run_result[len_orig_run_ops+i]
                    i += 1

                del run_result
                #wae_lambda = 0.0001*np.abs(rec_grad_np) / np.abs(global_grad_np)

                if zxz_lambda != 0.0:
                    _, zxz_loss_np = self.sess.run([self.zxz_opt, self.zxz_loss], feed_dict={self.zxz_lambda: zxz_lambda, self.sample_noise: batch_noise, self.lr_decay: decay, self.is_training: True})
                    if 'NEPTUNE_API_TOKEN' in os.environ:
                        neptune.send_metric('loss_zxz', x=counter, y=zxz_loss_np)
                
                # grads = self.sess.run(self.grad_extra, feed_dict={self.sample_noise: batch_noise, self.is_training: True})
                # for el in grads:
                #    print  el

                #a = np.dot(P_np, np.ones(P_np.shape[0]))
                #print(a)
                #b = np.dot(np.transpose(P_np), np.ones(P_np.shape[1]))
                #print(b)
                # Update the adversary in Z space for WAE-GAN

                if opts['z_test'] == 'gan':
                    loss_adv = self.loss_gan[0]
                    _ = self.sess.run(
                        [self.z_adv_opt, loss_adv],
                        feed_dict={self.sample_points: batch_images,
                                   self.sample_noise: batch_noise,
                                   self.wae_lambda: wae_lambda,
                                   self.rec_lambda: rec_lambda,
                                   self.lr_decay: decay,
                                   self.is_training: True})
               
                # del batch_images

                self.recalculate_x_latents(data, self.train_size, opts['batch_size'], overwrite_placeholder=True, ids=all_data_ids)


                self.x_rec_losses_np[data_ids] = per_sample_rec_loss_np
                if self.sparsifier is not None:
                    self.sparsifier.on_batch_end()

                # Update learning rate if necessary

                if opts['lr_schedule'] == "plateau":
                    # First 30 epochs do nothing
                    if epoch >= 30:
                        # If no significant progress was made in last 10 epochs
                        # then decrease the learning rate.
                        if loss < min(losses[-20 * batches_num:]):
                            wait = 0
                        else:
                            wait += 1
                        if wait > 10 * batches_num:
                            decay = max(decay  / 1.4, 1e-6)
                            logging.error('Reduction in lr: %f' % decay)
                            wait = 0

                losses.append(loss)
                losses_rec.append(loss_rec)
                losses_match.append(loss_match)
                if opts['verbose']:
                    logging.error('Matching penalty after %d steps: %f' % (
                        counter, losses_match[-1]))


                if 'NEPTUNE_API_TOKEN' in os.environ:
                    for k, v in run_result_dict.items():
                        neptune.send_metric(k, x=counter, y=v)

                    neptune.send_metric('loss_wae_matching', x=counter, y=loss_match)
                    neptune.send_metric('loss_rec', x=counter, y=loss_rec)
                    neptune.send_metric('loss', x=counter, y=loss)
                    neptune.send_metric('stay_loss', x=counter, y=stay_loss)
                    neptune.send_metric('wae_lambda', x=counter, y=wae_lambda)
                    neptune.send_metric('rec_lambda', x=counter, y=rec_lambda)
                    neptune.send_metric('lr', x=counter, y=decay)
                    #neptune.send_metric('rec_grad_np', x=counter, y=rec_grad_np)
                    #neptune.send_metric('global_grad_np', x=counter, y=global_grad_np)

                # Update regularizer if necessary
                if opts['lambda_schedule'] == 'adaptive':
                    if wait_lambda >= 999 and len(losses_rec) > 0:
                        last_rec = losses_rec[-1]
                        last_match = losses_match[-1]
                        wae_lambda = 0.5 * wae_lambda + \
                                     0.5 * last_rec / abs(last_match)
                        if opts['verbose']:
                            logging.error('Lambda updated to %f' % wae_lambda)
                        wait_lambda = 0
                    else:
                        wait_lambda += 1

                """
                if opts['z_test_scope'] == 'global':
                    sample_qz = self.x_latents_with_current_batch
                    sample_pz = self.nat_targets
                else:
                    sample_qz = self.encoded
                    sample_pz = self.sample_noise
                """
                
                grad = tf.gradients(self.sinkhorn_loss(self.x_latents, self.nat_targets), self.x_latents)
                #grad = tf.gradients(self.x_latents, self.x_latents)
                grads_of_latents = np.asarray(self.sess.run(
                    grad, feed_dict = feed_d)[0])
                """
                gradients_of_first_batch = np.asarray(self.sess.run(
                    tf.gradients(self.sinkhorn_loss(self.x_latents, self.nat_targets), self.x_latents[:batch_size]),
                    feed_dict = feed_d)[0])
                """
                pos_of_latents = np.asarray(self.sess.run(self.x_latents, feed_dict = feed_d))
                grads_and_pos = np.concatenate((grads_of_latents, pos_of_latents), axis = 1)
                def my_print(arr, f):
                    for line in arr:
                        print("\t".join(map(str, line)), file=f)
                my_print(grads_and_pos, f=gradlog_file)
                gradlog_file.flush()

                proj_grads_of_latents = grads_of_latents[:, :2]
                proj_pos_of_latents = pos_of_latents[:, :2]
                proj_current_batch = proj_pos_of_latents[it * batch_size : (it + 1) * batch_size]
                
                if counter > 0:
                    fig, ax = plt.subplots()
                    ax.scatter(x = prev_proj_pos_of_latents[:, 0], y = prev_proj_pos_of_latents[:, 1], s = 20, c = 'g')
                    ax.scatter(x = prev_proj_current_batch[:, 0], y = prev_proj_current_batch[:, 1], s = 20, c = 'm')
                    ax.scatter(x = proj_pos_of_latents[:, 0], y = proj_pos_of_latents[:, 1], s = 10, c = 'b')
                    proj_move = proj_pos_of_latents - prev_proj_pos_of_latents
                    ax.quiver(prev_proj_pos_of_latents[:, 0], prev_proj_pos_of_latents[:, 1],
                              prev_proj_grads_of_latents[:, 0], prev_proj_grads_of_latents[:, 1],
                              angles = 'xy', scale_units = 'xy', scale = 1, width = 0.001)
                    ax.quiver(prev_proj_pos_of_latents[:, 0], prev_proj_pos_of_latents[:, 1],
                             proj_move[:, 0], proj_move[:, 1], color = "c",
                              angles = 'xy', scale_units = 'xy', scale = 1, width = 0.001)
                    plt.savefig(os.path.join(opts["work_dir"] + str(counter - 1) + "_pos_latents.png"), dpi=200)
                    plt.close()

                prev_proj_grads_of_latents = np.copy(proj_grads_of_latents)
                prev_proj_pos_of_latents = np.copy(proj_pos_of_latents)           
                prev_proj_current_batch = np.copy(proj_current_batch)

                counter += 1

                # Print debug info

                if counter % opts['print_every'] == 0:
                    now = time.time()

                    # Auto-encoding test images

                    [loss_rec_test, enc_test, rec_test] = self.sess.run(
                        [self.loss_reconstruct, self.encoded, self.reconstructed],
                        feed_dict={self.sample_points: data.test_data[:self.num_pics],
                                   self.is_training: False})

                    if enc_test_prev is not None:
                        changes = np.mean((enc_test - enc_test_prev) ** 2.)
                        encoding_changes.append(changes)
                    else:
                        changes = np.mean((enc_test) ** 2.)
                        encoding_changes.append(changes)

                    enc_test_prev = enc_test

                    nat_size = self.opts['nat_size']
                    assert len(data.test_data) >= nat_size
                    test_latents = self.encode(data.test_data[:nat_size], opts['batch_size'])
                    test_targets = self.sample_pz(nat_size)

                    global_sinkhorn_loss = self.sess.run(self.sinkhorn_loss_tf,
                        feed_dict={self.latents_ph: test_latents,
                                   self.targets_ph: test_targets,
                                   self.is_training: False})


                    # Auto-encoding training images

                    [loss_rec_train, enc_train, rec_train] = self.sess.run(
                        [self.loss_reconstruct, self.encoded, self.reconstructed],
                        feed_dict={self.sample_points: data.data[:self.num_pics],
                                   self.is_training: False})

                    # Random samples generated by the model

                    sample_gen = self.sess.run(
                        self.decoded,
                        feed_dict={self.sample_noise: self.fixed_noise,
                                   self.is_training: False})

                    # Blurriness measures

                    gen_blurr = self.sess.run(
                        self.blurriness,
                        feed_dict={self.sample_points: sample_gen})
                    blurr_vals.append(np.min(gen_blurr))

                    # Printing various loss values

                    debug_str = 'EPOCH: %d/%d, BATCH:%d/%d, BATCH/SEC:%.2f' % (
                        epoch + 1, opts['epoch_num'],
                        it + 1, batches_num,
                        float(counter) / (now - self.start_time))
                    debug_str += ' (WAE_LOSS=%.5f, RECON_LOSS=%.5f, ' \
                                 'MATCH_LOSS=%.5f, ' \
                                 'RECON_LOSS_TEST=%.5f, ' \
                                 'SHARPNESS=%.5f)' % (
                                    losses[-1], losses_rec[-1],
                                    losses_match[-1], loss_rec_test, np.min(gen_blurr))
                    logging.error(debug_str)

                    # Printing debug info for encoder variances if applicable
                    if opts['e_noise'] == 'gaussian':
                        logging.error('Per dimension encoder variances:')
                        per_dim_range = self.debug_sigmas.eval(
                            session = self.sess,
                            feed_dict={self.sample_points: data.test_data[:500],
                                       self.is_training: False})
                        for idim in range(per_dim_range.shape[0]):
                            if per_dim_range[idim][1] > 0.:
                                logging.error(
                                    'dim%.4d: [%.2f; %.2f; %.2f]  <------' % (idim,
                                       per_dim_range[idim][0],
                                       per_dim_range[idim][2],
                                       per_dim_range[idim][1]))
                            else:
                                logging.error(
                                    'dim%.4d: [%.2f; %.2f; %.2f]' % (idim,
                                       per_dim_range[idim][0],
                                       per_dim_range[idim][2],
                                       per_dim_range[idim][1]))

                    # Choosing the 2d projection for Pz vs Qz plots
                    pz_noise = self.sample_pz(opts['plot_num_pics'])
                    if opts['pz'] == 'normal' and opts['zdim'] > 2:
                        # Finding the least Gaussian projection for Qz
                        proj_mat, check = self.least_gaussian_2d(
                            np.vstack([enc_train, enc_test]))
                        # Projecting samples from Qz and Pz on this 2d plain
                        Qz_train = np.dot(enc_train, proj_mat)
                        Qz_test = np.dot(enc_test, proj_mat)
                        Pz = np.dot(pz_noise, proj_mat)
                        nat_targets_proj = np.dot(self.nat_targets_np, proj_mat)

                    else:
                        Qz_train = enc_train[:, :2]
                        Qz_test = enc_test[:, :2]
                        Pz = pz_noise[:, :2]
                        nat_targets_proj = self.nat_targets_np[:, :2]

                    # Making plots

                    summary_plot, transport_plot = save_plots(opts, data.data[:self.num_pics],
                               data.test_data[:self.num_pics],
                               rec_train[:self.num_pics],
                               rec_test[:self.num_pics],
                               sample_gen,
                               Qz_train, Qz_test, Pz, nat_targets_proj,
                               losses_rec, losses_match, blurr_vals,
                               encoding_changes,
                               'res_e%04d_mb%05d.png' % (epoch, it)
                               #, P_np
                               )

                    generated_batches = []
                    for l in range(10000//batch_size):
                        noise = self.sample_pz(batch_size)
                        sample_gen = self.sess.run(self.decoded, feed_dict={self.sample_noise: noise, self.is_training: False})
                        generated_batches.append(sample_gen)
                    generated = np.concatenate(generated_batches, axis=0)

                    synthetic = opts['dataset'].startswith('syn') or opts['dataset'].startswith('checkers')
                    if synthetic:
                        plot_dicts = plot_syn.get_plots(generated, opts, counter-1)
                        for plot_dict in plot_dicts:
                            if 'NEPTUNE_API_TOKEN' in os.environ:
                                pass # neptune.send_image(plot_dict['name'], x=counter-1, y=plot_dict['plot'])

                    if 'NEPTUNE_API_TOKEN' in os.environ:
                        neptune.send_metric('rec_loss_test', x=counter-1, y=loss_rec_test)
                        neptune.send_metric('blurriness', x=counter-1, y=np.min(gen_blurr))
                        neptune.send_metric('global_ot_loss', x=counter-1, y=global_sinkhorn_loss)
                        #neptune.send_image('transport_plot', transport_plot)
                        print("skipping sending image, issues with neptune")
                        # neptune.send_image('summary_plot', summary_plot)

        # Save the final model
        video.close()
        gradlog_file.close()
        if True:#epoch > 0:
            self.saver.save(self.sess,
                             os.path.join(opts['work_dir'],
                                          'checkpoints',
                                          'trained-wae-final'),
                             global_step=counter)

    def add_sigmas_debug(self):

        # Ops to debug variances of random encoders
        enc_sigmas = self.enc_sigmas
        enc_sigmas = tf.Print(
            enc_sigmas,
            [tf.nn.top_k(tf.reshape(enc_sigmas, [-1]), 1).values[0]],
            'Maximal log sigmas:')
        enc_sigmas = tf.Print(
            enc_sigmas,
            [-tf.nn.top_k(tf.reshape(-enc_sigmas, [-1]), 1).values[0]],
            'Minimal log sigmas:')
        self.enc_sigmas = enc_sigmas

        enc_sigmas_t = tf.transpose(self.enc_sigmas)
        max_per_dim = tf.reshape(tf.nn.top_k(enc_sigmas_t, 1).values, [-1, 1])
        min_per_dim = tf.reshape(-tf.nn.top_k(-enc_sigmas_t, 1).values, [-1, 1])
        avg_per_dim = tf.reshape(tf.reduce_mean(enc_sigmas_t, 1), [-1, 1])
        per_dim = tf.concat([min_per_dim, max_per_dim, avg_per_dim], axis=1)
        self.debug_sigmas = per_dim

def save_plots(opts, sample_train, sample_test,
               recon_train, recon_test,
               sample_gen,
               Qz_train, Qz_test, Pz, nat_targets,
               losses_rec, losses_match, blurr_vals,
               encoding_changes,
               filename
               #, P_np
               ):
    """ Generates and saves the plot of the following layout:
        img1 | img2 | img3
        img4 | img6 | img5

        img1    -   test reconstructions
        img2    -   train reconstructions
        img3    -   samples
        img4    -   Qz vs Pz plots
        img5    -   real pics
        img6    -   loss curves

    """
    num_pics = opts['plot_num_pics']
    num_cols = opts['plot_num_cols']
    assert num_pics % num_cols == 0
    assert num_pics % 2 == 0
    greyscale = sample_train.shape[-1] == 1

    if opts['input_normalize_sym']:
        sample_train = sample_train / 2. + 0.5
        sample_test = sample_test / 2. + 0.5
        recon_train = recon_train / 2. + 0.5
        recon_test = recon_test / 2. + 0.5
        sample_gen = sample_gen / 2. + 0.5

    images = []

    # Reconstruction plots
    for pair in [(sample_train, recon_train),
                 (sample_test, recon_test)]:

        # Arrange pics and reconstructions in a proper way
        sample, recon = pair
        assert len(sample) == num_pics
        assert len(sample) == len(recon)
        pics = []
        merged = np.vstack([recon, sample])
        r_ptr = 0
        w_ptr = 0
        for _ in range(num_pics // 2):
            merged[w_ptr] = sample[r_ptr]
            merged[w_ptr + 1] = recon[r_ptr]
            r_ptr += 1
            w_ptr += 2

        for idx in range(num_pics):
            if greyscale:
                pics.append(1. - merged[idx, :, :, :])
            else:
                pics.append(merged[idx, :, :, :])

        # Figuring out a layout
        pics = np.array(pics)
        image = np.concatenate(np.split(pics, num_cols), axis=2)
        image = np.concatenate(image, axis=0)
        images.append(image)

    # Sample plots
    for sample in [sample_gen, sample_train]:

        assert len(sample) == num_pics
        pics = []
        for idx in range(num_pics):
            if greyscale:
                pics.append(1. - sample[idx, :, :, :])
            else:
                pics.append(sample[idx, :, :, :])

        # Figuring out a layout
        pics = np.array(pics)
        image = np.concatenate(np.split(pics, num_cols), axis=2)
        image = np.concatenate(image, axis=0)
        images.append(image)

    img1, img2, img3, img5 = images

    # Creating a pyplot fig
    dpi = 100
    height_pic = img1.shape[0]
    width_pic = img1.shape[1]
    fig_height = 4 * height_pic / float(dpi)
    fig_width = 6 * width_pic / float(dpi)

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = matplotlib.gridspec.GridSpec(2, 3)

    # Filling in separate parts of the plot

    # First samples and reconstructions
    for img, (gi, gj, title) in zip([img1, img2, img3, img5],
                             [(0, 0, 'train reconstruction'),
                              (0, 1, 'test reconstruction'),
                              (0, 2, 'generated samples'),
                              (1, 2, 'data points')]):
        plt.subplot(gs[gi, gj])
        if greyscale:
            image = img[:, :, 0]
            # in Greys higher values correspond to darker colors
            ax = plt.imshow(image, cmap='Greys',
                            interpolation='none', vmin=0., vmax=1.)
        else:
            ax = plt.imshow(img, interpolation='none', vmin=0., vmax=1.)

        ax = plt.subplot(gs[gi, gj])
        plt.text(0.47, 1., title,
                 ha="center", va="bottom", size=30, transform=ax.transAxes)

        # Removing ticks
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.set_xlim([0, width_pic])
        ax.axes.set_ylim([height_pic, 0])
        ax.axes.set_aspect(1)

    # Then the Pz vs Qz plot
    ax = plt.subplot(gs[1, 0])

    plt.scatter(nat_targets[:, 0], nat_targets[:, 1],
                color='black', s=2, marker='+', label='Targets')
    plt.scatter(Pz[:, 0], Pz[:, 1],
                color='red', s=70, marker='*', label='Pz')
    plt.scatter(Qz_train[:, 0], Qz_train[:, 1], color='blue',
                s=20, marker='x', edgecolors='face', label='Qz_train')
    plt.scatter(Qz_test[:, 0], Qz_test[:, 1], color='green',
                s=20, marker='x', edgecolors='face', label='Qz_test')
    plt.text(0.47, 1., 'Pz vs Qz plot',
             ha="center", va="bottom", size=30, transform=ax.transAxes)
    xmin = min(np.min(Qz_train[:,0]),
               np.min(Qz_test[:,0]))
    xmax = max(np.max(Qz_train[:,0]),
               np.max(Qz_test[:,0]))
    magnify = 0.3
    width = abs(xmax - xmin)
    xmin = xmin - width * magnify
    xmax = xmax + width * magnify

    ymin = min(np.min(Qz_train[:,1]),
               np.min(Qz_test[:,1]))
    ymax = max(np.max(Qz_train[:,1]),
               np.max(Qz_test[:,1]))
    width = abs(ymin - ymax)
    ymin = ymin - width * magnify
    ymax = ymax + width * magnify
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.legend(loc='upper left')

    # The loss curves
    ax = plt.subplot(gs[1, 1])
    total_num = len(losses_rec)
    x_step = max(total_num // 100, 1)
    x = np.arange(1, len(losses_rec) + 1, x_step)

    y = np.log(np.abs(losses_rec[::x_step]))
    plt.plot(x, y, linewidth=2, color='red', label='log(|rec loss|)')

    y = np.log(np.abs(losses_match[::x_step]))
    plt.plot(x, y, linewidth=2, color='blue', label='log(|match loss|)')

    blurr_mod = np.tile(blurr_vals, (opts['print_every'], 1))
    blurr_mod = blurr_mod.transpose().reshape(-1)
    x_step = max(len(blurr_mod)// 100, 1)
    x = np.arange(1, len(blurr_mod) + 1, x_step)
    y = np.log(blurr_mod[::x_step])
    plt.plot(x, y, linewidth=2, color='orange', label='log(sharpness)')
    if len(encoding_changes) > 0:
        x = np.arange(1, len(losses_rec) + 1)
        y = np.log(encoding_changes)
        x_step = len(x) // len(y)
        plt.plot(x[::x_step], y, linewidth=2, color='green', label='log(encoding changes)')
    plt.grid(axis='y')
    plt.legend(loc='upper right')

    # Saving
    utils.create_dir(opts['work_dir'])
    fig.savefig(utils.o_gfile((opts["work_dir"], filename), 'wb'),
                dpi=dpi, format='png')

    buffer = io.StringIO()
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    summary_plot = PIL.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())

    plt.clf()
    #plt.imshow(P_np, cmap='hot', interpolation='nearest')
    #plt.savefig(os.path.join(opts["work_dir"],filename+".P.png"))

    buffer = io.StringIO()
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    transport_plot = PIL.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
    fig.clear()   
    plt.close()

    return (summary_plot, transport_plot)
