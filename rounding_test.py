import numpy as np
import tensorflow as tf

import sinkhorn

#F = tf.constant([[1,2,10,3,5],[3,8,5,1,2],[100,1,2,7,5],[1,2,3,4,5],[1,2,3,4,5]], np.float64)
F = tf.random_uniform([5,4], minval=0.5, maxval=100, dtype=tf.float64)
#c = tf.ones(4, np.float64)
#r = tf.ones(5, np.float64)

r = tf.constant([1/5,1/5,1/5,1/5,1/5], tf.float64)
c = tf.constant([1/4,1/4,1/4,1/4], tf.float64)


with tf.Session() as sess:
    print(F.eval())
    G1 = (sess.run(sinkhorn.rounding(F, r, c)))
    rowsum = sess.run(tf.math.reduce_sum(G1, axis = 1))
    colsum = sess.run(tf.math.reduce_sum(G1, axis = 0))

print(G1)
print(rowsum)
print(colsum)
