import numpy as np
import tensorflow as tf

from sinkhorn import *
from sparse import *


def broadcast_test():
    with tf.Session() as sess:
        def e(t):
            return sess.run(t)
        def p(s, t):
            print(s, e(t))

        n = 100000
        np.random.seed(0)
        sparse_values = tf.Variable(np.random.normal(size=2).astype(np.float32))
        dense_vector_np = np.arange(n) + 1
        dense_vector = tf.broadcast_to(tf.cast(tf.constant(dense_vector_np), np.float32), (n,))
        s = tf.sparse.SparseTensor(indices=[[0, 0], [2, 3]], values=sparse_values, dense_shape=[n, n])

        e(tf.global_variables_initializer())

        p("sparse mat:", s)
        print("dense vec:", dense_vector_np)

        for axis in (0, 1):
            print("==================")
            print("axis =", axis)

            res = sparse_matrix_dense_broadcasted_vector_add(s, dense_vector, axis=axis)
            p("broadcasted elementwise add:", res)

            s_np_trunc = np.zeros((4, 4)) ; s_np_trunc[0, 0] = e(sparse_values[0]) ; s_np_trunc[2, 3] = e(sparse_values[1])
            dense_vector_trunc = dense_vector_np[np.newaxis, :4]
            if axis == 0:
                masked_add = (s_np_trunc + dense_vector_trunc.T) * (s_np_trunc != 0)
            elif axis == 1:
                masked_add = (s_np_trunc + dense_vector_trunc) * (s_np_trunc != 0)
            print("numpy result truncated:", masked_add)

            p("gradients", tf.gradients(tf.sparse.reduce_sum(res), sparse_values))


def trunc_test():
    with tf.Session() as sess:
        def e(t):
            return sess.run(t)
        def p(s, t):
            print(s, e(t))

        x_np = np.random.normal(size=(5, 4)).astype(np.float32)
        y_np = np.random.normal(size=(3, 4)).astype(np.float32)
        x = tf.constant(x_np)
        y = tf.Variable(y_np)
        e(tf.global_variables_initializer())

        dense = pdist(x, y)

        sparse = SparsePdist(x, y, rows=5, cols=3, k=4)

        print("x", x_np, "y", y_np)
        p("dense dist", dense)
        p("sparse dist", to_dense(sparse))

        sparse_summed = tf.sparse.sparse_dense_matmul(sparse, tf.ones((3, 1)))
        p("sparse_summed", sparse_summed)
        grad = tf.gradients(sparse_summed, [y])
        p("sparse_summed grad by y", grad)

        dense_summed = tf.matmul(dense, tf.ones((3, 1)))
        p("dense_summed", dense_summed)
        dense_grad = tf.gradients(dense_summed, [y])
        p("dense_summed grad by y", dense_grad)


def sparse_full_sinkhorn_test():
    with tf.Session() as sess:
        def e(t):
            return sess.run(t)
        def p(s, t):
            print(s, e(t))

        n = 5
        m = 3
        d = 4
        k = n - 1
        epsilon = 0.01
        niter = 10

        np.random.seed(4)
        x_np = np.random.normal(size=(n, d)).astype(np.float32) / 10
        y_np = np.random.normal(size=(m, d)).astype(np.float32) / 10
        x = tf.constant(x_np)
        y = tf.Variable(y_np)

        C_sparse = SparsePdist(x, y, rows=n, cols=m, k=k)
        C_dense = to_dense(C_sparse)
        C_dense = tf.where(tf.equal(C_dense, 0.0), np.inf * tf.ones_like(C_dense), C_dense)

        e(tf.global_variables_initializer())

        p("C_sparse", C_sparse)
        p("C_dense", C_dense)

        f_sparse = tf.linspace(0.1, 0.2, n)
        f_dense = f_sparse
        for i in range(niter):
            f_sparse, g_sparse = SparseSinkhorn_step(C_sparse, f_sparse, epsilon)
            f_dense, g_dense = Sinkhorn_step(C_dense, f_dense, epsilon)
            print(i)
            p("f_sparse", f_sparse)
            p("f_dense", f_dense)
            p("g_sparse", g_sparse)
            p("g_dense", g_dense)


def sparse_sinkhorn_test():
    with tf.Session() as sess:
        def e(t):
            return sess.run(t)
        def p(s, t):
            print(s, e(t))

        n = 5
        m = 3
        d = 4
        k = n - 1
        epsilon = 1.0

        np.random.seed(4)
        x_np = np.random.normal(size=(n, d)).astype(np.float32) / 10
        y_np = np.random.normal(size=(m, d)).astype(np.float32) / 10
        x = tf.constant(x_np)
        y = tf.Variable(y_np)

        if True:
            # some random sparse
            dense = tf.constant(np.random.binomial(1, 0.5, size=(n, m)) * np.random.normal(size=(n, m)), dtype=tf.float32)
            C = to_sparse(dense)
            dense = tf.where(tf.equal(dense, 0.0), np.inf * tf.ones_like(dense), dense)
        elif False:
            # a zero matrix but sparsely represented
            dense = tf.constant(np.random.binomial(1, 0.5, size=(n, m)), dtype=tf.float32)
            C = to_sparse(dense)
            C = scalar_mul(C, 0.0)
            dense = tf.zeros_like(dense)
        else:
            C = SparsePdist(x, y, rows=n, cols=m, k=k)
            dense = to_dense(C)
            dense = tf.where(tf.equal(dense, 0.0), np.inf * tf.ones_like(dense), dense)

        e(tf.global_variables_initializer())

        p("C sparse", C)
        p("C dense", dense)

        '''

        f = tf.linspace(0.1, 0.2, n)
        g = tf.linspace(0.1, 0.2, m)

        p("translation0", -g)
        p("translation1", -f)

        translated0 = sparse_matrix_dense_broadcasted_vector_add(C, -g, axis=0)
        translated1 = sparse_matrix_dense_broadcasted_vector_add(C, -f, axis=1)

        p("translated0 dense", dense - g)
        # p("translated0 sparse", translated0)
        p("translated0 sparse to_dense", to_dense(translated0))

        p("translated1 dense", tf.transpose(tf.transpose(dense) - f))
        # p("translated1 sparse", translated1)
        p("translated1 sparse to_dense", to_dense(translated1))


        '''
        print("===============")
        f_init = tf.linspace(0.1, 0.2, n)

        g = epsilon * tf.reduce_logsumexp((-f_init - tf.transpose(dense)) / epsilon, -1)
        f = epsilon * tf.reduce_logsumexp((-g - dense) / epsilon, -1)

        p("translated dense", tf.transpose(-f_init - tf.transpose(dense)))
        p("g dense", g)
        p("translated2 dense", (-g - dense))
        p("f dense", f)

        def td(s):
            return to_dense(s)

        '''
        mx = 0.0 * tf.ones(C.dense_shape[0], dtype=tf.float32)
        p("add", td(sparse_matrix_dense_broadcasted_vector_add(C, -mx, 1)))
        p("lse", lse(sparse_matrix_dense_broadcasted_vector_add(C, -mx, 1)))
        p("full", lse(sparse_matrix_dense_broadcasted_vector_add(C, -mx, 1)) + mx)
        return
        '''

        translated = sparse_matrix_dense_broadcasted_vector_add(minus(C), -f_init, axis=1) # TODO or is it axis=0?

        p("translated", td(translated))
        p("td(scalar_mul(translated, 1.0/epsilon))", td(scalar_mul(translated, 1.0/epsilon)))
        p("lse", sparse_logsumexp(scalar_mul(translated, 1.0/epsilon), 0))
        return

        g = epsilon * sparse_logsumexp(scalar_mul(translated, 1.0/epsilon), 0)
        translated2 = sparse_matrix_dense_broadcasted_vector_add(minus(C), -g, axis=0) # TODO or is it axis=1?
        f = epsilon * sparse_logsumexp(scalar_mul(translated2, 1.0 / epsilon), 1)
        # p("translated sparse", translated)
        p("translated sparse_to_dense", to_dense(translated))
        p("g sparse", g)
        # p("translated2 sparse", translated2)
        p("translated2 sparse_to_dense", to_dense(translated2))
        p("f sparse", f)
        return



if __name__ == "__main__":
    sparse_sinkhorn_test()
    sparse_full_sinkhorn_test()
    trunc_test()
    broadcast_test()
