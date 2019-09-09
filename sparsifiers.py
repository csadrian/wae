import numpy as np
import itertools


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

    def __init__(self, sources, targets, k, sess):
        super().__init__(sources, targets)
        self.k = k
        self.sess = sess
        self.reinit()

    def reinit(self):
        pass
    
    def indices(self):
        
        return self._indices

    def on_batch_end(self):
        pass
