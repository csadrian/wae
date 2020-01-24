import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import cv2

from sparse import *


# squared pairwise euclidean distance matrix.
# NOTE: might be negative for numerical precision reasons.
def pdist(x, y):
    nx = tf.reduce_sum(tf.square(x), 1)
    ny = tf.reduce_sum(tf.square(y), 1)

    nx = tf.reshape(nx, [-1, 1])
    ny = tf.reshape(ny, [1, -1])

    # return (nx - 2*tf.matmul(x, y, False, True) + ny)
    # it used to be Wasserstein_1:
    sqrt_epsilon = 1e-9
    return tf.sqrt(tf.maximum(nx - 2*tf.matmul(x, y, False, True) + ny, sqrt_epsilon))


def pdist_more_memory_eaten_fewer_numerical_issues(x, y):
    dx = x[:, None, :] - y[None, :, :]
    return tf.reduce_sum(tf.square(dx), -1)


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


def Sinkhorn_step(C, f, epsilon):
    g = epsilon * tf.reduce_logsumexp((-f - tf.transpose(C)) / epsilon, -1)
    f = epsilon * tf.reduce_logsumexp((-g - C) / epsilon, -1)
    return f, g


def SparseSinkhorn_step(C, f, epsilon):
    translated = sparse_matrix_dense_broadcasted_vector_add(minus(C), -f, axis=1)
    g = epsilon * sparse_logsumexp(scalar_mul(translated, 1.0/epsilon), 0)
    translated2 = sparse_matrix_dense_broadcasted_vector_add(minus(C), -g, axis=0)
    f = epsilon * sparse_logsumexp(scalar_mul(translated2, 1.0 / epsilon), 1)
    return f, g


# TODO get rid of P_temp which only served debugging purposes
def Sinkhorn(C, f=None, epsilon=None, niter=10):
    assert epsilon is not None
    n = tf.shape(C)[0]
    if f is None:
        f = tf.zeros(n, np.float32)
    for i in range(niter):
        f, g = Sinkhorn_step(C, f, epsilon)

    P_temp = -f[:, None] - C
    P = (-f[:, None] - g[None, :] - C) / epsilon
    # P = rounding_log(P, tf.zeros(n, tf.float32), tf.zeros(n, tf.float32))
    OT = tf.reduce_sum(tf.exp(P) * C)
    return OT, P_temp, P, f, g


# TODO get rid of P_temp which only served debugging purposes
def SparseSinkhorn(C, f=None, epsilon=None, niter=10):
    assert epsilon is not None
    n, m = C.get_shape().as_list()
    if f is None:
        f = tf.zeros(n, np.float32)
    for i in range(niter):
        f, g = SparseSinkhorn_step(C, f, epsilon)

    # happy times when it looked this simple:
    # P = (-f[:, None] - g[None, :] - C) / epsilon
    P_temp = sparse_matrix_dense_broadcasted_vector_add(minus(C), -f, 1)
    P = P_temp
    P = sparse_matrix_dense_broadcasted_vector_add(P, -g, 0)
    P = scalar_mul(P, 1.0 / epsilon)
    OT = tf.reduce_sum(tf.exp(P.values) * C.values)
    return OT, P_temp, P, f, g


# TODO get rid of P_temp which only served debugging purposes
def SinkhornLoss(sources, targets, epsilon=0.01, niter=10):
    C = pdist(sources, targets)
    OT, P_temp, P, f, g = Sinkhorn(C, f=None, epsilon=epsilon, niter=niter)
    return OT, P_temp, P, f, g, C


def SparseSinkhornLoss(sources, targets, sparse_indices, epsilon=0.01, niter=10):
    sparse_xs = tf.gather(sources, sparse_indices[:, 0], validate_indices=False)
    sparse_ys = tf.gather(targets, sparse_indices[:, 1], validate_indices=False)
    # W1
    sparse_dists = tf.sqrt(1e-8 + tf.reshape(tf.reduce_sum(tf.square(sparse_xs-sparse_ys), -1), (-1, )))
    sparse_dist_matrix = tf.SparseTensor(sparse_indices, sparse_dists, (sources.get_shape().as_list()[0], targets.get_shape().as_list()[0]))
    C = sparse_dist_matrix
    OT, P_temp, P, f, g = SparseSinkhorn(C, f=None, epsilon=epsilon, niter=niter)
    return OT, P_temp, P, f, g, C


# SinkhornLoss corrected with autocorrelation
# TODO get rid of P_temp which only served debugging purposes
def SinkhornDivergence(sources, targets, epsilon=0.01, niter=10):
    OTxy, _, Pxy, fxy, gxy, Cxy = SinkhornLoss(sources, targets, epsilon=epsilon, niter=niter)
    OTxx, _, Pxx, fxx, gxx, Cxx = SinkhornLoss(sources, sources, epsilon=epsilon, niter=niter)
    OTyy, _, Pyy, fyy, gyy, Cyy = SinkhornLoss(targets, targets, epsilon=epsilon, niter=niter)
    return OTxy - 0.5 * (OTxx + OTyy), Pxy, fxy, gxy, Cxy


def rounding_linear_domain(F, r, c):
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


def rounding_log_domain(F, r, c):
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
        cv2.circle(img, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA, shift=0)
    return img


def draw_edges(p1, p2, w, radius, edges=True):
    img = np.zeros((w, w, 3), np.uint8)

    p1 = p1 / radius
    p2 = p2 / radius

    p1 = np.int32(w / 2 + p1[:, :2] * w / 2)
    p2 = np.int32(w / 2 + p2[:, :2] * w / 2)
    if edges:
        for (x1, y1), (x2, y2) in zip(p1, p2):
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
    for (x2, y2) in p2:
        cv2.circle(img, (x2, y2), 5, (0, 0, 255), -1, cv2.LINE_AA)
    for (x1, y1) in p1:
        cv2.circle(img, (x1, y1), 3, (255, 0, 0), -1, cv2.LINE_AA)
    return img
