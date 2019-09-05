import numpy as np
import tensorflow as tf

import sinkhorn

n = 5
m = 4

# F = tf.constant([[1,2,10,3,5],[3,8,5,1,2],[100,1,2,7,5],[1,2,3,4,5],[1,2,3,4,5]], np.float64)

np.random.seed(1)
F_np = np.random.uniform(size=(n, m), low=0.5, high=100).astype(np.float32)
F = tf.constant(F_np)

r = tf.ones(n, tf.float32) / n
c = tf.ones(m, tf.float32) / m

with tf.Session() as sess:
    def e(t):
        return sess.run(t)
    def p(s, t):
        print(s, e(t))

    print(F.eval())
    # G1 = sinkhorn.rounding(F, r, c)

    # p("G1", G1)

    F_log = tf.math.log(F)
    r_log = tf.math.log(r)
    c_log = tf.math.log(c)
    # p("F_log", F_log)
    # p("r_log", r_log)
    # p("c_log", c_log)

    result = sinkhorn.rounding(F, r, c)

    result_log = sinkhorn.rounding_log(F_log, r_log, c_log)
    result_explog = tf.math.exp(result_log)
    p("result", result)
    p("result_explog", result_explog)

    exit()

    rowsum = tf.math.reduce_sum(result, axis=1)
    colsum = tf.math.reduce_sum(result, axis=0)
    p("rowsum", rowsum)
    p("colsum", colsum)

    rowsum_explog = tf.math.reduce_sum(result_explog, axis=1)
    colsum_explog = tf.math.reduce_sum(result_explog, axis=0)
    p("rowsum_explog", rowsum_explog)
    p("colsum_explog", colsum_explog)

    exit()

    G1_log = sinkhorn.rounding_log(F_log, r_log, c_log)
    p("exp G1_log", tf.math.exp(G1_log))
    # p("rounding_log F_log", rounded)
