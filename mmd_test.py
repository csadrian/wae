import numpy as np
import tensorflow as tf

def mmd_penalty(sample_qz, sample_pz, pz_scale, kernel='RBF'):
    sigma2_p = pz_scale ** 2
    n, d = sample_pz.get_shape().as_list()
    n = tf.cast(n, tf.int32)
    nf = tf.cast(n, tf.float32)
    half_size = (n * n - n) / 2

    norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keep_dims=True)
    dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
    distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

    norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keep_dims=True)
    dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
    distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

    dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
    distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

    if kernel == 'RBF':
        # Median heuristic for the sigma^2 of Gaussian kernel
        '''
        sigma2_k = tf.nn.top_k(
            tf.reshape(distances, [-1]), half_size).values[half_size - 1]
        sigma2_k += tf.nn.top_k(
            tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
        '''
        # Maximal heuristic for the sigma^2 of Gaussian kernel
        # sigma2_k = tf.nn.top_k(tf.reshape(distances_qz, [-1]), 1).values[0]
        # sigma2_k += tf.nn.top_k(tf.reshape(distances, [-1]), 1).values[0]
        sigma2_k = d * sigma2_p
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
        pz_kind = 'normal'
        if pz_kind == 'normal':
            Cbase = 2. * d * sigma2_p
        elif pz_kind == 'sphere':
            Cbase = 2.
        elif pz_kind == 'uniform':
            # E ||x - y||^2 = E[sum (xi - yi)^2]
            #               = zdim E[(xi - yi)^2]
            #               = const * zdim
            Cbase = d
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            res1 = C / (C + distances_qz)
            res1 += C / (C + distances_pz)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = C / (C + distances)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat += res1 - res2
    else:
        assert False

    return stat


def main():
    with tf.Session() as sess:
        def e(t):
            return sess.run(t)
        def p(s, t):
            print(s, e(t))

        n = 10000
        d = 64
        scale = tf.Variable(1.0, dtype=tf.float32)
        sample_qz = scale * tf.random.normal((n, d), dtype=tf.float32)
        sample_pz = tf.random.normal((n, d), dtype=tf.float32)
        mmd = mmd_penalty(sample_qz, sample_pz, pz_scale=1.0, kernel='IMQ')
        e(tf.global_variables_initializer())
        for scale_np in np.linspace(-2, +2, 21):
            print(scale_np, sess.run(mmd, feed_dict={scale: scale_np}))


if __name__ == "__main__":
    main()
