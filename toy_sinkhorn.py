import tensorflow as tf
import numpy as np
import sinkhorn
import ops
from collections import OrderedDict

# Functions needed for the model

def lrelu(x, leak=0.3):
    return tf.maximum(x, leak * x)

def batch_norm(opts, _input, is_train, reuse, scope, scale=True):

    return tf.contrib.layers.batch_norm(
        opts, _input, center=True, scale=scale,
        epsilon=1e-05, decay=0.9,
        is_training=is_train, reuse=reuse, updates_collections=None,
        scope=scope, fused=False)

def linear(opts, input_, output_dim, scope=None, init='normal', reuse=None):
    stddev = opts['init_std']
    bias_start = opts['init_bias']
    shape = input_.get_shape().as_list()

    assert len(shape) > 0
    in_shape = shape[1]
    if len(shape) > 2:
        input_ = tf.reshape(input_, [-1, np.prod(shape[1:])])
        in_shape = np.prod(shape[1:])

    with tf.variable_scope(scope or "lin", reuse=reuse):
        if init == 'normal':
            matrix = tf.get_variable(
                "W", [in_shape, output_dim], tf.float32,
                tf.random_normal_initializer(stddev=stddev))
        else:
            matrix = tf.get_variable(
                "W", [in_shape, output_dim], tf.float32,
                tf.constant_initializer(np.identity(in_shape)))
        bias = tf.get_variable(
            "b", [output_dim],
            initializer=tf.constant_initializer(bias_start))
        
    return tf.matmul(input_, matrix) + bias

# Models 

def encoder(opts, inputs, reuse=False, is_training=False):
    def add_noise(x):
        shape = tf.shape(x)
        return x + tf.truncated_normal(shape, 0.0, 0.01)
    def do_nothing(x):
        return x
    inputs = tf.cond(is_training,
                     lambda: add_noise(inputs), lambda: do_nothing(inputs))

    e_num_units = 10
    e_num_layers = 2

    with tf.variable_scope("encoder", reuse=reuse):
        hi = inputs
        i = 0
        for i in range(e_num_layers):
            hi = linear(opts, hi, e_num_units, scope='h%d_lin' % i)
            if opts['batch_norm']:
                hi = batch_norm(opts, hi, is_training,
                                    reuse, scope='h%d_bn' % i)
            hi = tf.nn.relu(hi)
            mean = linear(opts, hi, opts['zdim'], 'mean_lin')
            log_sigmas = linear(opts, hi, opts['zdim'], 'log_sigmas_lin')
            res = (mean, log_sigmas)
                    
        noise_matrix = None

        if opts['pz'] == 'uniform':
            res = tf.nn.tanh(res)

        return res, noise_matrix

class simple_encoder(object):
    def __init__(self, opts, train_size=0):

        self.sess = tf.Session()
        self.opts = opts
        self.train_size = train_size
        self.nat_pos = 0

        self.tensors_to_log = OrderedDict()

        # -- Some of the parameters for future use

        self.data_shape = [64, 64, 3]

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
        self.enc_mean, self.enc_sigmas = None, None
        
        # -- Objectives, losses, penalties

        self.add_nat_tensors()
        self.penalty, self.loss_gan = self.matching_penalty()
        self.latents_ph = tf.placeholder(tf.float32, shape=(opts['nat_size'], opts['zdim']))
        self.targets_ph = tf.placeholder(tf.float32, shape=(opts['nat_size'], opts['zdim']))
        self.sinkhorn_loss_tf = self.sinkhorn_loss(self.latents_ph, self.targets_ph)
        self.wae_objective = self.wae_lambda * self.penalty

        # Extra costs if any
        
        self.add_least_gaussian2d_ops()

        # -- Optimizers, savers, etc

        self.add_optimizers()
        self.add_savers()
        self.init = tf.global_variables_initializer()


    def add_nat_placeholders(self):
        opts = self.opts
        self.nat_targets_np = self.sample_pz(self.opts['nat_size'])
        self.nat_targets = tf.Variable(self.nat_targets_np, dtype=tf.float32, trainable=False)
        self.x_latents = tf.Variable(tf.zeros(opts['nat_size'], opts['zdim']), dtype=tf.float32, trainable=False)
        self.batch_indices_mod = tf.placeholder(tf.int64, shape=(opts['batch_size'],))

    def add_nat_tensors(self):
        opts = self.opts
        n = opts['nat_size']
        bs = opts['batch_size']

        x_latents_with_current_batch = tf.stop_gradient(tf.boolean_mask(self.x_latents,
            tf.sparse_to_dense(
                sparse_indices=self.batch_indices_mod,
                default_value=1.0,
                sparse_values=0.0,
                output_shape=[n], validate_indices=False
                )
            ))
        x_latents_with_current_batch = tf.concat([x_latents_with_current_batch, 
            self.encoded[:bs]], axis=0)
        x_latents_with_current_batch = tf.reshape(x_latents_with_current_batch, shape=(n, self.opts['zdim']))
        self.x_latents_with_current_batch = x_latents_with_current_batch

        self.nat_targets_update_ph = tf.placeholder(self.nat_targets.dtype, shape=self.nat_targets.get_shape())
        self.update_nat_targets_op = self.nat_targets.assign(self.nat_targets_update_ph)

        self.x_latents_update_ph = tf.placeholder(self.x_latents.dtype, shape=self.x_latents.get_shape())
        self.update_x_latents_op = self.x_latents.assign(self.x_latents_update_ph)

        self.su_ids_to_update_ph = tf.placeholder(tf.int64, shape=(self.opts['recalculate_size'],))
        self.su_latents_ph = tf.placeholder(tf.float32, shape=(self.opts['recalculate_size'], self.opts['zdim']))
        self.scatter_update_x_latents_op = tf.scatter_update(self.x_latents, self.su_ids_to_update_ph, self.su_latents_ph)

    def resample_nat_targets(self):
        self.nat_targets_np = self.sample_pz(opts['nat_size'])
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
        #zxz_lambda = tf.placeholder(tf.float32, name='zxz_lambda_ph')

        self.lr_decay = decay
        self.wae_lambda = wae_lambda
        self.is_training = is_training

    def sinkhorn_loss(self, sample_qz, sample_pz):
        opts = self.opts
        decayed_epsilon = tf.constant(opts['sinkhorn_epsilon'])
        OT, P_temp, P, f, g, C = sinkhorn.SinkhornLoss(sample_qz, sample_pz, epsilon=decayed_epsilon, niter=opts['sinkhorn_iters'])
        return OT

    def add_savers(self):
        opts = self.opts
        saver = tf.train.Saver(max_to_keep=10)
        tf.add_to_collection('real_points_ph', self.sample_points)
        tf.add_to_collection('noise_ph', self.sample_noise)
        tf.add_to_collection('is_training_ph', self.is_training)
        tf.add_to_collection('encoder', self.encoded)
        self.saver = saver

    def matching_penalty(self):
        opts = self.opts
        if opts['z_test_scope'] == 'global':
            sample_qz = self.x_latents_with_current_batch
            sample_pz = self.nat_targets
        else:
            sample_qz = self.encoded
            sample_pz = self.sample_noise

        loss_match = self.sinkhorn_loss(sample_qz, sample_pz)
        self.add_to_log("sinkhorn_ot", loss_match)
        return loss_match, loss_gan

    def optimizer(self, lr, decay=1.):
        opts = self.opts
        lr = opts['lr']
        lr *= decay
        if opts['optimizer'] == "sgd":
            return tf.train.GradientDescentOptimizer(lr)
        elif opts['optimizer'] == "adam":
            return tf.train.AdamOptimizer(lr, beta1=opts['adam_beta1'])
        else:
            assert False, 'Unknown optimizer.'

    def add_optimizers(self):
        encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        self.pretrain_opt = None
        
    def sample_pz(self, num=100):
        opts = self.opts
        noise = None
        distr = opts['pz']
        if distr == 'uniform':
            noise = np.random.uniform(
                -1, 1, [num, opts['zdim']]).astype(np.float32)
        elif distr in ('normal', 'sphere'):
            mean = np.zeros(opts['zdim'])
            cov = np.identity(opts['zdim'])
            noise = np.random.multivariate_normal(
                mean, cov, num).astype(np.float32)
            if distr == 'sphere':
                noise = noise / np.sqrt(
                    np.sum(noise * noise, axis=1))[:, np.newaxis]
        return opts['pz_scale'] * noise

    def encode(self, x, batch_size):
        latents_list = []
        for k in range(x.shape[0] // batch_size):
            batch_images_temp = x[k*batch_size:(k+1)*batch_size]
            batch_latents_temp, = self.sess.run([self.encoded], feed_dict={self.is_training: False, self.sample_points: batch_images_temp})
            latents_list.append(batch_latents_temp)
        latents = np.concatenate(latents_list, axis=0)
        return latents

    def recalculate_x_latents(self, data, train_size, batch_size, overwrite_placeholder=True, ids=None):
        opts = self.opts
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


    
    def train(self, data):
        opts = self.opts
        logging.error('Training WAE')
        losses = []
        losses_match = []
        encoding_changes = []
        enc_test_prev = None
        batches_num = self.train_size // opts['batch_size']

        self.sess.run(self.init)

        self.start_time = time.time()
        counter = 0
        decay = 1.
        wae_lambda = opts['lambda']
        batch_size = opts['batch_size']

        # Variables for dynamic updates
        wait = 0
        wait_lambda = 0

        gradlog_file = open("gradlog.txt", "w")
        
        VIDEO_SIZE = 512
        with FFMPEG_VideoWriter(opts['name'] + 'out.mp4', (VIDEO_SIZE, VIDEO_SIZE), 3.0) as video:
          #if True:

          self.recalculate_x_latents(data, self.train_size, batch_size, overwrite_placeholder=True, ids=None)

          for epoch in range(opts['epoch_num']):

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
                self.saver.save(self.sess, os.path.join(opts['work_dir'], 'checkpoints', 'trained-wae'),
                                global_step=counter)

            # Iterate over batches

            if self.opts['nat_resampling'] == 'epoch':
                self.resample_nat_targets()

            for it in range(batches_num):

                if self.opts['nat_resampling'] == 'batch':
                    self.resample_nat_targets()

                # Sample batches of data points and Pz noise
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
                    if opts['pz'] == 'normal':
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
                    self.is_training: True,
                    self.batch_indices_mod: data_ids_mod}

                #ot_grads_and_vars_np = self.sess.run([self.ot_grads_and_vars], feed_dict=feed_d)

                run_ops = [self.wae_objective, self.penalty]
                len_orig_run_ops = len(run_ops)
                for key, value in self.get_tensors_to_log().items():
                    run_ops.append(value)

                run_result = self.sess.run(run_ops, feed_dict=feed_d)
                [loss, loss_match] = run_result[:len_orig_run_ops]

                run_result_dict = {}
                i = 0
                for key, value in self.get_tensors_to_log().items():
                    run_result_dict[key] = run_result[len_orig_run_ops+i]
                    i += 1

                del run_result

                self.recalculate_x_latents(data, self.train_size, batch_size, overwrite_placeholder=True, ids = None)

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
                losses_match.append(loss_match)

                if 'NEPTUNE_API_TOKEN' in os.environ:
                    for k, v in run_result_dict.items():
                        neptune.send_metric(k, x=counter, y=v)

                    neptune.send_metric('loss_wae_matching', x=counter, y=loss_match)
                    neptune.send_metric('loss', x=counter, y=loss)
                    neptune.send_metric('wae_lambda', x=counter, y=wae_lambda)
                    neptune.send_metric('lr', x=counter, y=decay)

                # Update regularizer if necessary
                wait_lambda += 1
                
                grad = tf.gradients(self.sinkhorn_loss(self.x_latents, self.nat_targets), self.x_latents)
                #grad = tf.gradients(self.x_latents, self.x_latents)
                grads_of_latents = np.asarray(self.sess.run(
                    grad, feed_dict = feed_d)[0])

                
                #nat_targets_np = self.sess.run(self.nat_targets, feed_dict = feed_d)
                #proj_nat_targets = nat_targets_np[:, :2]
        
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
                    #ax.scatter(x = proj_nat_targets[:, 0], y = proj_nat_targets[:, 1], s = 10, c = 'y')
                    ax.scatter(x = prev_proj_pos_of_latents[:, 0], y = prev_proj_pos_of_latents[:, 1], s = 20, c = 'g')
                    ax.scatter(x = prev_proj_current_batch[:, 0], y = prev_proj_current_batch[:, 1], s = 20, c = 'm')
                    ax.scatter(x = proj_pos_of_latents[:, 0], y = proj_pos_of_latents[:, 1], s = 5, c = 'b')
                    proj_move = proj_pos_of_latents - prev_proj_pos_of_latents
                    ax.quiver(prev_proj_pos_of_latents[:, 0], prev_proj_pos_of_latents[:, 1],
                              prev_proj_grads_of_latents[:, 0], prev_proj_grads_of_latents[:, 1],
                              angles = 'xy', scale_units = 'xy', scale = 1, width = 0.001)
                    ax.quiver(prev_proj_pos_of_latents[:, 0], prev_proj_pos_of_latents[:, 1],
                             proj_move[:, 0], proj_move[:, 1], color = "c",
                              angles = 'xy', scale_units = 'xy', scale = 1, width = 0.001)
                    plt.savefig(os.path.join(opts['work_dir'] + str(counter - 1) + "_pos_latents.png"), dpi=200)
                    plt.close()

                prev_proj_grads_of_latents = np.copy(proj_grads_of_latents)
                prev_proj_pos_of_latents = np.copy(proj_pos_of_latents)           
                prev_proj_current_batch = np.copy(proj_current_batch)

                counter += 1

                # Print debug info

                if counter % opts['print_every'] == 0:
                    now = time.time()

                    # Auto-encoding test images

                    enc_test = self.sess.run(self.encoded,
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

                    enc_train = self.sess.run(self.encoded,
                        feed_dict={self.sample_points: data.data[:self.num_pics],
                                   self.is_training: False})

                    # Random samples generated by the model

                    debug_str = 'EPOCH: %d/%d, BATCH:%d/%d, BATCH/SEC:%.2f' % (
                        epoch + 1, opts['epoch_num'],
                        it + 1, batches_num,
                        float(counter) / (now - self.start_time))
                    debug_str += ' (WAE_LOSS=%.5f, MATCH_LOSS=%.5f)' % (
                                    losses[-1], losses_match[-1])
                    logging.error(debug_str)

                    # Printing debug info for encoder variances if applicable

                    if 'NEPTUNE_API_TOKEN' in os.environ:
                        neptune.send_metric('global_ot_loss', x=counter-1, y=global_sinkhorn_loss)
                        #neptune.send_image('transport_plot', transport_plot)
                        print("skipping sending image, issues with neptune")
                        # neptune.send_image('summary_plot', summary_plot)

        # Save the final model
        video.close()
        gradlog_file.close()
        if True:
            self.saver.save(self.sess, os.path.join(opts['work_dir'], 'checkpoints', 'trained-wae-final'),
                             global_step=counter)

            
