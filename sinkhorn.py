import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import cv2
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter


def mat_vec_fn(x, y_i, rows, cols, i, k, sess):
    v = tf.reduce_sum(tf.square(x-y_i), axis=1)
    top_values, top_indices = tf.nn.top_k(-v, k=k)
    top_indices = tf.expand_dims(top_indices, 1)
    temp_indices = tf.zeros((k,), dtype=tf.int32) + i*tf.ones((k,), dtype=tf.int32)
    temp_indices = tf.expand_dims(temp_indices, 1)
    top_indices = tf.concat([top_indices, temp_indices], 1)
    return tf.SparseTensor(tf.cast(top_indices, tf.int64), -top_values, dense_shape=(rows, cols))

# TODO
# 1. this is matmul, rather than ||x_i - y_j||^2.
# 2. when K_alpha, we need to get rid of the 1s as -inf is supposed to be the default.
def sparse_k_alpha(x, y, k, rows, cols, sess):

    def mat_vec_fn_closure(y_i, i):
        return mat_vec_fn(x, y_i, rows, cols, i, k, sess)
    spliced = [mat_vec_fn_closure(y[:, i], i) for i in range(cols)]

    res = spliced[0]
    print("res0", res.eval(session=sess))

    for i in range(1, cols):
        res = tf.sparse.add(res, spliced[i])
    return res
    # result = SparseTensor(input.indices, map_fn(fn, input.values), input.dense_shape)


def trunc_test():
    with tf.Session() as sess:
        x_np = np.random.normal(size=(5, 4)).astype(np.float32)
        y_np = np.random.normal(size=(4, 3)).astype(np.float32)
        x = tf.constant(x_np)
        y = tf.Variable(y_np)
        sess.run(tf.global_variables_initializer())

        sparse = sparse_k_alpha(x, y, k=2, rows=5, cols=3, sess=sess)

        print("x", x_np, "y", y_np)
        print("np", pdist(x, tf.transpose(y)).eval(session=sess))
        print(sess.run(tf.sparse.to_dense(sparse, validate_indices=False)).T)

        sparse_summed = tf.sparse.sparse_dense_matmul(sparse, tf.ones((3, 1)))
        print(sess.run(sparse_summed))
        g = tf.gradients(sparse_summed, [y])
        print(sess.run(g))
        return
        print(sess.run(tf.sparse.to_dense(sparse, validate_indices=False)).T)
        print(sess.run(x).dot(sess.run(y)))
        print(sess.run(g))




# squared pairwise euclidean distance matrix.
# NOTE: might be negative for numerical precision reasons.
def pdist(x,y):
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


# TODO that reduce_mean is some constant multiplier away from the standard value.
def Sinkhorn(C, f=None, epsilon=None, niter=10):
    assert epsilon is not None
    n = tf.shape(C)[0]
    if f is None:
        f = tf.zeros(n, np.float32)
    for i in range(niter):
        f, g = Sinkhorn_step(C, f, epsilon)

    P = (-f[:, None] - g[None, :] - C) / epsilon
    OT = tf.reduce_mean(tf.exp(P) * C)
    return OT, P, f, g


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
    trunc_test()
