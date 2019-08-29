import numpy as np
import tensorflow as tf
import cv2
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter


def slices_to_dims(slice_indices):
  """
  Args:
    slice_indices: An [N, k] Tensor mapping to column indices.
  Returns:
    An index Tensor with shape [N * k, 2], corresponding to indices suitable for
    passing to SparseTensor.
  """
  slice_indices = tf.cast(slice_indices, tf.int64)
  num_rows = tf.shape(slice_indices, out_type=tf.int64)[0]
  row_range = tf.range(num_rows)
  item_numbers = slice_indices * num_rows + tf.expand_dims(row_range, axis=1)
  item_numbers_flat = tf.reshape(item_numbers, [-1])
  return tf.stack([item_numbers_flat % num_rows,
                   item_numbers_flat // num_rows], axis=1)


# Every row zeroed except top k elements of row.
def zeroed_except_top_k_no_gradient(dense_matrix, k):
    dense_shape = dense_matrix.get_shape()
    top_values, top_indices = tf.nn.top_k(dense_matrix, k=k)
    sparse_indices = slices_to_dims(top_indices)
    sparse_tensor = tf.sparse_reorder(tf.SparseTensor(
        indices=sparse_indices,
        values=tf.reshape(top_values, [-1]),
        dense_shape=dense_shape))
    densified_top = tf.sparse_tensor_to_dense(sparse_tensor)
    return densified_top


# Every element set to large number except smallest k elements of each column.
def infd_except_bottom_k(dense_matrix, k):
    dense_matrix = tf.transpose(dense_matrix) # we want columns now
    dense_shape = dense_matrix.get_shape()
    rows, columns = dense_shape
    top_values, top_indices = tf.nn.top_k(dense_matrix, k=rows-k)
    sparse_indices = slices_to_dims(top_indices)
    sparse_tensor = tf.sparse_reorder(tf.SparseTensor(
        indices=sparse_indices,
        values=tf.ones_like(tf.reshape(top_values, [-1])),
        dense_shape=dense_shape))
    densified_top = tf.sparse_tensor_to_dense(sparse_tensor)
    merged = tf.where(densified_top > 0, 1000 * tf.ones_like(dense_matrix) , dense_matrix)
    merged = tf.transpose(merged) # we reverse the transpose that we've started with.
    return merged


def trunc_test():
    with tf.Session() as sess:
        a = tf.constant(np.random.normal(size=(6, 6)).astype(np.float32))
        b = sess.run(infd_except_bottom_k(a, 3))
        print(b)


def distmat(x,y):
    nx = tf.reduce_sum(tf.square(x), 1)
    ny = tf.reduce_sum(tf.square(y), 1)

    # na as a row and nb as a column vectors
    nx = tf.reshape(nx, [-1, 1])
    ny = tf.reshape(ny, [1, -1])

    # return pairwise euclidean difference matrix
    sqrt_epsilon = 0.0000000001
    d = tf.maximum(tf.sqrt(tf.maximum(nx - 2*tf.matmul(x, y, False, True) + ny, 0.0)) -  0.001, 0.0)
    return d


def sparse_distmat(x,y):
    nx = tf.reduce_sum(tf.square(x), 1)
    ny = tf.reduce_sum(tf.square(y), 1)

    # na as a row and nb as a column vectors
    nx = tf.reshape(nx, [-1, 1])
    ny = tf.reshape(ny, [1, -1])

    # return pairwise euclidean difference matrix
    sqrt_epsilon = 0.0000000001
    d = tf.sqrt(tf.maximum(nx - 2*tf.matmul(x, y, False, True) + ny, sqrt_epsilon))
    idx = tf.where(tf.less(d, 0.1))
    sparse = tf.SparseTensor(idx, tf.gather_nd(d, idx), d.get_shape())
    return sparse



def pdist(x, y):
    print("MESSED THINGS UP, EXPERIMENTING")
    keep_k = 50
    return infd_except_bottom_k(distmat(x, y), keep_k)

    return distmat(x, y)
    dx = x[:, None, :] - y[None, :, :]
    return tf.reduce_sum(tf.square(dx), -1)# / (0.0001)


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
    g = epsilon * tf.reduce_logsumexp((-f-tf.transpose(C)) / epsilon, -1)
    f = epsilon * tf.reduce_logsumexp((-g-C) / epsilon, -1)
    return f, g


def Sinkhorn(C, n, m, f=None, epsilon=None, niter=10):
    assert epsilon is not None
    n_t = tf.shape(C)[0]
    if f is None:
        f = tf.zeros(n_t, np.float32)
    for i in range(niter):
        f, g = Sinkhorn_step(C, f, epsilon)
    P = tf.exp((-f[:, None]-g[None, :]-C) / epsilon) / tf.cast(n, tf.float32)
    return P, f, g


def Sinkhorn_log_domain(C, n, m, f=None, epsilon=None, niter=10):
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
    for (x1, y1), (x2, y2) in zip(p1, p2):
        cv2.circle(img, (x2, y2), 5, (0, 0, 255), -1, cv2.CV_AA)
    for (x1, y1), (x2, y2) in zip(p1, p2):
        cv2.circle(img, (x1, y1), 3, (255, 0, 0), -1, cv2.CV_AA)
    return img
