import numpy as np
import tensorflow as tf
import cv2
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter





def distmat(x,y):
    nx = tf.reduce_sum(tf.square(x), 1)
    ny = tf.reduce_sum(tf.square(y), 1)
    
    # na as a row and nb as a co"lumn vectors
    nx = tf.reshape(nx, [-1, 1])
    ny = tf.reshape(ny, [1, -1])

    # return pairwise euclidead difference matrix
    #return tf.sqrt(tf.maximum(nx - 2*tf.matmul(x, y, False, True) + ny, 0.0)) ** 4  # / 0.0001
    return tf.maximum(tf.sqrt(tf.maximum(nx - 2*tf.matmul(x, y, False, True) + ny, 0.0)) -  0.001, 0.0)
    
def pdist(x, y):
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
    a = tf.ones((n,1))/float(n)
    b = tf.ones((1,m))/float(m)

    n_t = tf.shape(C)[0]

    if f is None:
        f = tf.zeros(n_t, np.float32)
    for i in range(niter):
        f, g = Sinkhorn_step(C, f, epsilon)

        #g = mina(C-f[:,None],epsilon,a)
        #f = minb(C-g[None,:],epsilon,b)
        # generate the coupling
        #P = a * tf.exp((f[:,None]+g[None,:]-C)/epsilon) * b
        # check conservation of mass
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
    p = np.int32(w / 2 + p * w / 4)
    # print(p)
    for x, y in p:
        cv2.circle(img, (x, y), 2, (255, 255, 255), -1, cv2.CV_AA, shift=0)
    return img


def draw_edges(p1, p2, w):
    img = np.zeros((w, w, 3), np.uint8)
    p1 = np.int32(w / 2 + p1 * w / 4)
    p2 = np.int32(w / 2 + p2 * w / 4)
    for (x1, y1), (x2, y2) in zip(p1, p2):
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
    for (x1, y1), (x2, y2) in zip(p1, p2):
        cv2.circle(img, (x2, y2), 5, (0, 0, 255), -1, cv2.CV_AA)
    for (x1, y1), (x2, y2) in zip(p1, p2):
        cv2.circle(img, (x1, y1), 3, (255, 0, 0), -1, cv2.CV_AA)
    return img

