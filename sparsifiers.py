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
        self.pointer_ph = tf.placeholder(tf.int64, shape=())
        xs = tf.slice(self.sources, [self.pointer_ph*self.batch_size, 0], [self.batch_size, self.sources.get_shape().as_list()[1]])
        ys = self.targets
        d = sinkhorn.pdist(xs, ys)
        self.top_values, self.top_indices = tf.nn.top_k(-d, k=5)
    
    def indices(self):
        all_indices = []        
        for i in range(self.n // self.batch_size):
            indices_np = np.arange(i*self.batch_size, (i+1)*self.batch_size)
            top_indices_np, = self.sess.run([self.top_indices], feed_dict={self.pointer_ph: i})
            top_indices_np = np.expand_dims(top_indices_np, axis=-1)
            ran = np.arange(i*self.batch_size, (i+1)*self.batch_size)
            temp = np.zeros_like(top_indices_np) + ran[:,None,None]
            top_indices_joined = np.concatenate([top_indices_np, temp], axis=2)
            top_indices_joined = np.reshape(top_indices_joined, (-1, 2))
            all_indices.append(top_indices_joined)
        indices = np.concatenate(all_indices, axis=0)
        return indices

    def on_batch_end(self):
        pass
