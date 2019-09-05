import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import cv2
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter


# that's a weird add, because it only works for the sparse subset of
# the matrix elements. the justification is that elsewhere the
# default is plus or minus infinity (rather than the standard 0), and inf + finite == inf.
#
# when axis == 0, we add v to each column of s,
# when axis == 1, we add v to each row of s.
def sparse_matrix_dense_broadcasted_vector_add(s, v, axis):
    return tf.SparseTensor(s.indices, tf.gather_nd(v, tf.reshape(s.indices[:, axis], (-1, 1))) + s.values, s.dense_shape)


def sparse_elementwise_op(s, op):
    return tf.SparseTensor(s.indices, op(s.values), s.dense_shape)


# TODO are these really not implemented?
def minus(s):
    return tf.SparseTensor(s.indices, -s.values, s.dense_shape)
def scalar_mul(s, f):
    return tf.SparseTensor(s.indices, f * s.values, s.dense_shape)


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


# creates the sparse vector that is the smallest k squared distances
# between x and y_i.
def mat_vec_fn(x, y_i, rows, cols, i, k):
    v = tf.reduce_sum(tf.square(x-y_i), axis=1)
    top_values, top_indices = tf.nn.top_k(-v, k=k)
    top_indices = tf.expand_dims(top_indices, 1)
    temp_indices = tf.zeros((k,), dtype=tf.int32) + i*tf.ones((k,), dtype=tf.int32)
    temp_indices = tf.expand_dims(temp_indices, 1)
    top_indices = tf.concat([top_indices, temp_indices], 1)
    return tf.SparseTensor(tf.cast(top_indices, tf.int64), -top_values, dense_shape=(rows, cols))


# keeps the first k elements of each row of the pairwise squared distance matrix
# x: (rows, d), y: (cols, d), returns (rows, cols)-dense_shaped
# sparse matrix with k sparse elements in each column.
#
# for example, if x are target points and y are latent points, each y is only affected
# by the closest k target points.
# BEWARE: when doing arithmetics with the result, you should
# assume that the non-instantiated elements are +inf, rather than the usual 0.
#
# TODO remove rows and cols args, they are redundant afaik. cols must be a python int, though.
# TODO unlike this, the rest of the code operates under the assumption that targets are the second arg.
#      this is a big issue because of the asymmetry that we want k friends for every latent
#      rather than k friends for every target.
def SparsePdist(x, y, rows, cols, k):
    y = tf.transpose(y)

    def mat_vec_fn_closure(y_i, i):
        return mat_vec_fn(x, y_i, rows, cols, i, k)
    spliced = [mat_vec_fn_closure(y[:, i], i) for i in range(cols)]

    res = spliced[0]

    for i in range(1, cols):
        res = tf.sparse.add(res, spliced[i])
    return res


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
        p("sparse dist", tf.sparse.to_dense(sparse, validate_indices=False))

        sparse_summed = tf.sparse.sparse_dense_matmul(sparse, tf.ones((3, 1)))
        p("sparse_summed", sparse_summed)
        grad = tf.gradients(sparse_summed, [y])
        p("sparse_summed grad by y", grad)

        dense_summed = tf.matmul(dense, tf.ones((3, 1)))
        p("dense_summed", dense_summed)
        dense_grad = tf.gradients(dense_summed, [y])
        p("dense_summed grad by y", dense_grad)

        return


# squared pairwise euclidean distance matrix.
# NOTE: might be negative for numerical precision reasons.
def pdist(x, y):
    nx = tf.reduce_sum(tf.square(x), 1)
    ny = tf.reduce_sum(tf.square(y), 1)

    nx = tf.reshape(nx, [-1, 1])
    ny = tf.reshape(ny, [1, -1])

    return (nx - 2*tf.matmul(x, y, False, True) + ny)
    # it used to be Wasserstein_1:
    # sqrt_epsilon = 1e-9
    # return tf.sqrt(tf.maximum(nx - 2*tf.matmul(x, y, False, True) + ny, sqrt_epsilon))


def pdist_more_memory_eaten_fewer_numerical_issues(x, y):
    dx = x[:, None, :] - y[None, :, :]
    return tf.reduce_sum(tf.square(dx), -1)


def Sinkhorn_step_nonent(C, f):
    g = tf.reduce_logsumexp(-f-tf.transpose(C), -1)
    f = tf.reduce_logsumexp(-g-C, -1)
    return f, g


def Sinkhorn_nonent(C, f=None, niter=1000):
    n = tf.shape(C)[0]
    if f is None:
        f = tf.zeros(n, np.float32)
    for i in range(niter):
        f, g = Sinkhorn_step(C, f)
    P = tf.exp(-f[:,None]-g[None,:]-C)/tf.cast(n, tf.float32)
    return P, f, g


def Sinkhorn_step(C, f, epsilon):
    g = epsilon * tf.reduce_logsumexp((-f - tf.transpose(C)) / epsilon, -1)
    f = epsilon * tf.reduce_logsumexp((-g - C) / epsilon, -1)
    return f, g


def sparse_reduce_sum(s, axis):
    n, m = s.get_shape().as_list()
    if axis == 0:
        # TODO seriously, no tf.sparse.dense_sparse_matmul()?
        summed = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(s), tf.ones((n, 1)))
    elif axis == 1 or axis == -1:
        summed = tf.sparse.sparse_dense_matmul(s, tf.ones((m, 1)))
    else:
        raise Exception("unimplemented")
    return summed


def sparse_exp(s):
    return sparse_elementwise_op(s, lambda e: tf.exp(e))


def sparse_reduce_min(s, axis):
    return - tf.sparse.reduce_max(minus(s), axis)


def sparse_logsumexp(s, axis):
    def lse(s):
        return tf.reshape(tf.log(sparse_reduce_sum(sparse_exp(s), axis)), [-1])

    # stupid function has no gradient:
    # mx = tf.sparse.reduce_max(s, axis)

    # attempt with mean but it did not help:
    # mx = tf.sparse.reduce_sum(s, axis) / tf.cast(s.dense_shape[axis], tf.float32)

    if axis == 0:
        other_axis = 1
    elif axis == 1 or axis == -1:
        other_axis = 0

    mx = 0.0 * tf.ones(s.dense_shape[other_axis], dtype=tf.float32)

    return lse(sparse_matrix_dense_broadcasted_vector_add(s, -mx, other_axis)) + mx


# TODO numerical stability went out of the window.
def my_sleazy_logsumexp(s, axis):
    n, m = s.get_shape().as_list()
    exped = sparse_elementwise_op(s, tf.exp)
    if axis == 0:
        # TODO seriously, no tf.sparse.dense_sparse_matmul()?
        summed = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(exped), tf.ones((n, 1)))
        logged = tf.log(summed)
        logged = tf.reshape(logged, (m, ))
    elif axis == 1 or axis == -1:
        summed = tf.sparse.sparse_dense_matmul(exped, tf.ones((m, 1)))
        logged = tf.log(summed)
        logged = tf.reshape(logged, (n, ))
    return logged


def to_sparse(dense):
    # Find indices where the tensor is not zero
    idx = tf.where(tf.not_equal(dense, 0))
    # Make the sparse tensor
    # Use tf.shape(a_t, out_type=tf.int64) instead of a_t.get_shape()
    # if tensor shape is dynamic
    sparse = tf.SparseTensor(idx, tf.gather_nd(dense, idx), tf.shape(dense, out_type=tf.int64))
    return sparse


def SparseSinkhorn_step(C, f, epsilon):
    translated = sparse_matrix_dense_broadcasted_vector_add(minus(C), -f, axis=1) # TODO or is it axis=0?
    g = epsilon * sparse_logsumexp(scalar_mul(translated, 1.0/epsilon), 0)
    translated2 = sparse_matrix_dense_broadcasted_vector_add(minus(C), -g, axis=0) # TODO or is it axis=1?
    f = epsilon * sparse_logsumexp(scalar_mul(translated2, 1.0 / epsilon), 1)
    return f, g


# TODO that reduce_mean is some constant multiplier away from the standard value.
def Sinkhorn(C, f=None, epsilon=None, niter=10):
    assert epsilon is not None
    n = tf.shape(C)[0]
    if f is None:
        f = tf.zeros(n, np.float32)
    for i in range(niter):
        f, g = Sinkhorn_step(C, f, epsilon)

    P = (-f[:, None] - g[None, :] - C) / epsilon
    # P = rounding_log(P, tf.zeros(n, tf.float32), tf.zeros(n, tf.float32))
    OT = tf.reduce_mean(tf.exp(P) * C)
    return OT, P, f, g


def SparseSinkhorn(C, f=None, epsilon=None, niter=10):
    assert epsilon is not None
    n, m = C.get_shape().as_list()
    if f is None:
        f = tf.zeros(n, np.float32)
    for i in range(niter):
        f, g = SparseSinkhorn_step(C, f, epsilon)

    # happy times when it looked this simple:
    # P = (-f[:, None] - g[None, :] - C) / epsilon
    P = sparse_matrix_dense_broadcasted_vector_add(minus(C), -f, 0) # TODO or is it the other way round?
    P = sparse_matrix_dense_broadcasted_vector_add(P, -g, 1)
    OT = tf.reduce_sum(tf.exp(P.values) * C.values)
    return OT, P, f, g


def sparse_sinkhorn_test():
    with tf.Session() as sess:
        def e(t):
            return sess.run(t)
        def p(s, t):
            print(s, e(t))

        n = 5
        m = 3
        d = 4
        k = n
        epsilon = 0.01

        np.random.seed(3)
        x_np = np.random.normal(size=(n, d)).astype(np.float32) / 10
        y_np = np.random.normal(size=(m, d)).astype(np.float32) / 10
        x = tf.constant(x_np)
        y = tf.Variable(y_np)
        sess.run(tf.global_variables_initializer())

        dense = pdist(x, y)

        C = SparsePdist(x, y, rows=n, cols=m, k=k)

        e(tf.global_variables_initializer())

        p("C", C)
        p("dense", dense)

        p("dense-sparse-dense", tf.sparse.to_dense(to_sparse(dense)))

        p("lse -C axis=0", sparse_logsumexp(minus(C), axis=0))
        p("lse -C axis=1", sparse_logsumexp(minus(C), axis=1))

        p("lse -dense axis=0", tf.reduce_logsumexp(-dense, axis=0))
        p("lse -dense axis=1", tf.reduce_logsumexp(-dense, axis=1))

        f = tf.zeros(n, np.float32)
        translated = sparse_matrix_dense_broadcasted_vector_add(minus(C), -f, axis=0) # TODO or is it axis=0?
        p("translated", translated)
        multiplied = scalar_mul(translated, 1.0/epsilon)
        p("multiplied", multiplied)
        p("sparse_logsumexp(multiplied, axis=0)", sparse_logsumexp(multiplied, axis=0))
        p("tf.logsumexp(tf.sparse.to_dense(multiplied), axis=0)",
            tf.reduce_logsumexp(tf.sparse.to_dense(multiplied, validate_indices=False), axis=0)
        )
        return
        g = epsilon * sparse_logsumexp(scalar_mul(translated, 1.0/epsilon), axis=0)
        p("g", g)
        translated2 = sparse_matrix_dense_broadcasted_vector_add(minus(C), -g, axis=1) # TODO or is it axis=1?
        p("translated2", translated2)
        f2 = epsilon * sparse_logsumexp(scalar_mul(translated2, 1.0 / epsilon), axis=1)
        p("f2", f2)


def Sinkhorn_log_domain(C, n, m, f=None, epsilon=None, niter=10):
    #raise Exception("please use Sinkhorn() instead")
    assert epsilon is not None
    a = tf.ones((n,1))/float(n)
    b = tf.ones((1,m))/float(m)

    def mina_u(H,epsilon): return -epsilon*tf.log( tf.reduce_sum(a * tf.exp(-H/epsilon),0) )
    def minb_u(H,epsilon): return -epsilon*tf.log( tf.reduce_sum(b * tf.exp(-H/epsilon),1) )

    def mina(H,epsilon): return mina_u(H-tf.reduce_min(H,0),epsilon) + tf.reduce_min(H,0);
    def minb(H,epsilon): return minb_u(H-tf.reduce_min(H,1)[:,None],epsilon) + tf.reduce_min(H,1);

    n_t = tf.shape(C)[0]

    if f is None:
        f = tf.zeros(n_t, np.float32)
    for i in range(niter):
        g = mina(C-f[:,None],epsilon)
        f = minb(C-g[None,:],epsilon)
        # generate the coupling
        #P = a * tf.exp((f[:,None]+g[None,:]-C)/epsilon) * b
        # check conservation of mass
        #Err[it] = np.linalg.norm(np.sum(P,0)-b,1)
    P = a * tf.exp((f[:,None]+g[None,:]-C)/epsilon) * b
    return P, f, g


# TODO evil hardwired constant.
# TODO correct terminology is unclear.
def SinkhornLoss(sources, targets, epsilon=0.01, niter=10):
    C = pdist(sources, targets)
    OT, P, f, g = Sinkhorn(C, f=None, epsilon=epsilon, niter=niter)
    return OT, P, f, g, C


def SparseSinkhornLoss(sources, targets, epsilon=0.01, niter=10, k=None):
    # rows = tf.shape(sources)[0]
    # cols = tf.shape(targets)[0]
    rows = sources.get_shape().as_list()[0]
    cols = targets.get_shape().as_list()[0]
    C = SparsePdist(sources, targets, rows, cols, k=k)
    OT, P, f, g = SparseSinkhorn(C, f=None, epsilon=epsilon, niter=niter)
    return OT, P, f, g, C


def EmulatedSparseSinkhornLoss(sources, targets, epsilon=0.01, niter=10):
    C = pdist(sources, targets)
    sh = C.get_shape()
    bernoulli = tfp.distributions.Bernoulli(probs=[0.9]) # prob of removal
    mask = tf.cast(bernoulli.sample(sh), dtype=np.float32)
    # sample() gives an extra axis for some reason.
    mask = mask[:, :, 0]
    C += mask * 1e9
    OT, P, f, g = Sinkhorn(C, f=None, epsilon=epsilon, niter=niter)
    return OT, P, f, g, C


# SinkhornLoss corrected with autocorrelation
def SinkhornDivergence(sources, targets, epsilon=0.01, niter=10):
    OTxy, Pxy, fxy, gxy, Cxy = SinkhornLoss(sources, targets, epsilon=epsilon, niter=niter)
    OTxx, Pxx, fxx, gxx, Cxx = SinkhornLoss(sources, sources, epsilon=epsilon, niter=niter)
    OTyy, Pyy, fyy, gyy, Cyy = SinkhornLoss(targets, targets, epsilon=epsilon, niter=niter)
    return OTxy - 0.5 * (OTxx + OTyy), Pxy, fxy, gxy, Cxy
    #return OTxy, Pxy, fxy, gxy, Cxy


def rounding(F, r, c):
    row_ones = tf.ones(tf.shape(F)[0],tf.float32)
    col_ones = tf.ones(tf.shape(F)[1],tf.float32)
    r_F = tf.math.reduce_sum(F, axis=1)
    #c_F = tf.math.reduce_sum(F, axis=0)
    X = tf.math.minimum(r/r_F, row_ones)
    DX = tf.diag(X)
    F1 = tf.matmul(DX,F)
    #r_F1 = tf.math.reduce_sum(F1, axis=1)
    c_F1 = tf.math.reduce_sum(F1, axis=0)
    Y = tf.math.minimum(c/c_F1, col_ones)
    DY = tf.diag(Y)
    F2 = tf.matmul(F1, DY)
    r_F2 = tf.math.reduce_sum(F2, axis=1)
    c_F2 = tf.math.reduce_sum(F2, axis=0)
    err_r = r - r_F2
    err_r = tf.expand_dims(err_r, 1)
    err_c = c - c_F2
    err_c = tf.expand_dims(err_c, 0)
    err_matrix = tf.matmul(err_r, err_c) / tf.norm(err_r, ord=1)
    G = F2 + err_matrix
    return G


def rounding_log(F, r, c):
    row_zeros = tf.zeros(tf.shape(F)[0],tf.float32)
    col_zeros = tf.zeros(tf.shape(F)[1],tf.float32)
    r_F = tf.math.reduce_logsumexp(F, axis=1)
    #c_F = tf.math.reduce_sum(F, axis=0)
    X = tf.math.minimum(r-r_F, row_zeros)
    X = tf.transpose(tf.broadcast_to(X, [tf.shape(F)[1], tf.shape(F)[0]]))
    F1 = X + F
    #r_F1 = tf.math.reduce_sum(F1, axis=1)
    c_F1 = tf.math.reduce_logsumexp(F1, axis=0)
    Y = tf.math.minimum(c-c_F1, col_zeros)
    Y = tf.broadcast_to(Y, [tf.shape(F)[0], tf.shape(F)[1]])
    F2 = Y + F1
    r_F2 = tf.math.reduce_logsumexp(F2, axis=1)
    c_F2 = tf.math.reduce_logsumexp(F2, axis=0)
    err_r = tf.math.log(tf.math.maximum(tf.exp(r) - tf.exp(r_F2), 0.0))

    # TODO expand_dims
    err_r_broadcasted = tf.transpose(tf.broadcast_to(err_r, (tf.shape(F)[1], tf.shape(F)[0])))

    err_c = tf.math.log(tf.math.maximum(tf.exp(c) - tf.exp(c_F2), 0.0))

    # TODO expand_dims
    err_c = tf.broadcast_to(err_c, (tf.shape(F)[0], tf.shape(F)[1]))
    err_matrix = err_r_broadcasted + err_c - tf.log(tf.norm(tf.exp(err_r), ord=1))
    G = tf.math.log(tf.exp(F2) + tf.exp(err_matrix))
    return G


def draw_points(p, w):
    img = np.zeros((w, w, 3), np.uint8)
    p = np.int32(w / 2 + p[:, :2] * w / 4)
    # print(p)
    for x, y in p:
        cv2.circle(img, (x, y), 2, (255, 255, 255), -1, cv2.CV_AA, shift=0)
    return img


def draw_edges(p1, p2, w, edges=True):
    img = np.zeros((w, w, 3), np.uint8)
    p1 = np.int32(w / 2 + p1[:, :2] * w / 8)
    p2 = np.int32(w / 2 + p2[:, :2] * w / 8)
    if edges:
        for (x1, y1), (x2, y2) in zip(p1, p2):
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
    for (x2, y2) in p2:
        cv2.circle(img, (x2, y2), 5, (0, 0, 255), -1, cv2.CV_AA)
    for (x1, y1) in p1:
        cv2.circle(img, (x1, y1), 3, (255, 0, 0), -1, cv2.CV_AA)
    return img

if __name__ == "__main__":
    sparse_sinkhorn_test() ; exit()
    trunc_test()
    broadcast_test()
