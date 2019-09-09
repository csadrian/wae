import numpy as np
import itertools
import tensorflow as tf

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
        self._full_indices = [k for k in itertools.product(np.arange(self.n, dtype=np.int32), repeat=2)]
    
    def indices(self):
        return self._full_indices


class RandomSparsifier(Sparsifier):

    def __init__(self, sources, targets, num_indices, opts={}):
        super().__init__(sources, targets)
        self.num_indices = num_indices
        self.reinit()

    def reinit(self):
        rnd_x = np.random.choice(self.n, size=(self.num_indices,1), replace=True)
        rnd_y = np.random.choice(self.m, size=(self.num_indices,1), replace=True)
        self._indices = np.concatenate([rnd_x, rnd_y], axis=1)
    
    def indices(self):
        return self._indices

    def on_batch_end(self):
        pass


class TfTopkSparsifier(Sparsifier):

    def __init__(self, sources, targets, k, sess, batch_size=100):
        super().__init__(sources, targets)
        self.k = k
        self.sess = sess
        self.batch_size = batch_size
        self.create_op()

    def create_op(self):
        #self.indices_ph = tf.placeholder(tf.int64, shape=(self.batch_size,))
        self.pointer_ph = tf.placeholder(tf.int64, shape=())
        self.bs = tf.constant(self.batch_size, dtype=tf.int64)
        xs = self.sources[self.pointer_ph*self.bs : (self.pointer_ph+1)*self.bs]
        ys = self.targets
        v = tf.reduce_sum(tf.square(xs-ys), axis=1)
        self.top_values, self.top_indices = tf.nn.top_k(-v, k=self.k)
    
    def indices(self):
        all_indices = []
        for i in range(self.n // self.batch_size):
            indices_np = np.arange(i*self.batch_size, (i+1)*self.batch_size)
            top_indices_np = self.sess.run([self.top_indices], feed_dict={self.pointer_ph: i})
            top_indices_np = top_indices_np[:,0] + (i*self.batch_size)
            all_indices.append(top_indices_np)
        indices = np.concatenate(all_indices, axis=0)
        return indices

    def on_batch_end(self):
        pass
