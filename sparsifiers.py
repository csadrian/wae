import numpy as np
import itertools
import tensorflow as tf
import sinkhorn


class Sparsifier():
    def __init__(self, sources, targets):
        self.sources = sources
        self.targets = targets
        self.n = sources.shape[0]
        self.m = targets.shape[0]

    def indices(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_batch_end(self):
        pass


class FullSparsifier(Sparsifier):
    def __init__(self, sources, targets):
        super().__init__(sources, targets)
        assert self.n == self.m, "Not implemented"
        self._full_indices = np.array([k for k in itertools.product(np.arange(self.n, dtype=np.int64), repeat=2)])

    def indices(self):
        return self._full_indices


class RandomSparsifier(Sparsifier):
    def __init__(self, n, m, num_indices, resample=True):
        self.n = n
        self.m = m
        self.num_indices = num_indices
        self.resample = resample
        self.reinit()

    def reinit(self):
        rnd_x = np.random.choice(self.n, size=(self.num_indices,1), replace=True)
        rnd_y = np.random.choice(self.m, size=(self.num_indices,1), replace=True)
        self._indices = np.concatenate([rnd_x, rnd_y], axis=1)
        """
        flat_indices = np.random.choice(self.n * self.m, self.num_indices, replace=False)
        self._indices = np.zeros((self.num_indices, 2)).astype(np.int64)
        self._indices[:, 0] = flat_indices % self.n
        self._indices[:, 1] = flat_indices // self.n
        """
    def indices(self):
        if self.resample:
            self.reinit()
        return self._indices

    def on_batch_end(self):
        pass


class TfTopkSparsifier(Sparsifier):
    def __init__(self, sources, targets, k, sess, batch_size=100):
        super().__init__(sources, targets)
        self.k = k
        self.sess = sess
        self.batch_size = batch_size
        assert self.n % batch_size == 0
        self.create_op()

    def create_op(self):
        self.pointer_ph = tf.placeholder(tf.int64, shape=())
        xs = tf.slice(self.sources, [self.pointer_ph*self.batch_size, 0], [self.batch_size, self.sources.get_shape().as_list()[1]])
        ys = self.targets
        d = sinkhorn.pdist(xs, ys)
        self.top_values, self.top_indices = tf.nn.top_k(-d, k=self.k)

    def indices(self):
        all_indices = []
        for i in range(self.n // self.batch_size):
            indices_np = np.arange(i*self.batch_size, (i+1)*self.batch_size)
            top_indices_np, = self.sess.run([self.top_indices], feed_dict={self.pointer_ph: i})
            top_indices_np = np.expand_dims(top_indices_np, axis=-1)
            ran = np.arange(i*self.batch_size, (i+1)*self.batch_size)
            temp = np.zeros_like(top_indices_np) + ran[:,None,None]
            top_indices_joined = np.concatenate([temp, top_indices_np], axis=2)
            top_indices_joined = np.reshape(top_indices_joined, (-1, 2))
            all_indices.append(top_indices_joined)
        indices = np.concatenate(all_indices, axis=0)
        return indices

    def on_batch_end(self):
        pass


class TfTwoWayTopkSparsifier(Sparsifier):
    def __init__(self, sources, targets, k, sess, batch_size=100):
        super().__init__(sources, targets)
        self.k = k
        self.sess = sess
        self.batch_size = batch_size
        assert self.n % batch_size == 0 and self.m % batch_size == 0
        self.create_op()

    def create_op(self):
        self.pointer_ph = tf.placeholder(tf.int64, shape=())
        xs = tf.slice(self.sources, [self.pointer_ph*self.batch_size, 0], [self.batch_size, self.sources.get_shape().as_list()[1]])
        ys = tf.slice(self.targets, [self.pointer_ph*self.batch_size, 0], [self.batch_size, self.targets.get_shape().as_list()[1]])
        d = sinkhorn.pdist(xs, self.targets)
        d_t = sinkhorn.pdist(self.sources, ys)
        self.top_values, self.top_indices = tf.nn.top_k(-d, k=self.k)
        self.top_values_t, self.top_indices_t = tf.nn.top_k(tf.transpose(-d_t), k=self.k)

    def indices(self):
        all_indices = []
        for i in range(self.n // self.batch_size):
            top_indices_np = self.sess.run(self.top_indices, feed_dict={self.pointer_ph: i})
            top_indices_np = np.expand_dims(top_indices_np, axis=-1)
            ran = np.arange(i*self.batch_size, (i+1)*self.batch_size)
            temp = np.zeros_like(top_indices_np) + ran[:,None,None]
            top_indices_joined = np.concatenate([temp, top_indices_np], axis=2)
            top_indices_joined = np.reshape(top_indices_joined, (-1, 2))
            all_indices.append(top_indices_joined)

        for i in range(self.n // self.batch_size):
            top_indices_t_np = self.sess.run(self.top_indices_t, feed_dict={self.pointer_ph: i})
            top_indices_t_np = np.expand_dims(top_indices_t_np, axis=-1)
            ran = np.arange(i*self.batch_size, (i+1)*self.batch_size)
            temp_t = np.zeros_like(top_indices_t_np) + ran[:,None,None]
            top_indices_t_joined = np.concatenate([top_indices_t_np, temp_t], axis=2)
            top_indices_t_joined = np.reshape(top_indices_t_joined, (-1, 2))
            all_indices.append(top_indices_t_joined)

        indices = np.concatenate(all_indices, axis=0)
        return indices

    def on_batch_end(self):
        pass


class SparsifierCombinator(Sparsifier):
    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2

    def indices(self):
        i1 = self.s1.indices()
        i2 = self.s2.indices()
        return np.unique(np.concatenate((i1, i2)), axis=0)


def sparsifier_test():
    n = 10
    d = 8
    num_indices = n * n // 5
    sources_np = np.random.normal(size=(n, d)).astype(np.float32) * 2 + 1
    targets_np = np.random.normal(size=(n, d)).astype(np.float32)
    # print(RandomSparsifier(sources_np, targets_np, num_indices).indices())

    with tf.Session() as sess:
        sources = tf.Variable(sources_np)
        targets = tf.Variable(targets_np)

        sess.run(tf.global_variables_initializer())

        k = 2
        tks = TfTwoWayTopkSparsifier(sources, targets, k, sess, batch_size=2)
        print(tks.indices())


if __name__ == "__main__":
    sparsifier_test()
