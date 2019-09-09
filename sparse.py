import tensorflow as tf

# that's a weird add, because it only works for the sparse subset of
# the matrix elements. the justification is that elsewhere the
# default is plus or minus infinity (rather than the standard 0), and inf + finite == inf.
#
# when axis == 0, we add v to each column of s,
# when axis == 1, we add v to each row of s.
def sparse_matrix_dense_broadcasted_vector_add(s, v, axis):
    assert axis in (0, 1)
    other_axis = 1 - axis
    return tf.SparseTensor(s.indices, tf.gather(v, s.indices[:, other_axis]) + s.values, s.dense_shape)


def sparse_elementwise_op(s, op):
    return tf.SparseTensor(s.indices, op(s.values), s.dense_shape)


# TODO are these really not implemented?
def minus(s):
    return tf.SparseTensor(s.indices, -s.values, s.dense_shape)


def scalar_mul(s, f):
    return tf.SparseTensor(s.indices, f * s.values, s.dense_shape)


def sparse_reduce_sum(s, axis):
    if axis == 0:
        # TODO seriously, no tf.sparse.dense_sparse_matmul()?
        summed = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(s), tf.ones((tf.shape(s)[0], 1)))
    elif axis == 1 or axis == -1:
        summed = tf.sparse.sparse_dense_matmul(s, tf.ones((tf.shape(s)[1], 1)))
    else:
        raise Exception("unimplemented")
    return summed


def sparse_exp(s):
    return sparse_elementwise_op(s, lambda e: tf.exp(e))


def sparse_reduce_min(s, axis):
    return - tf.sparse.reduce_max(minus(s), axis)


def sparse_naive_logsumexp(s, axis):
    return tf.reshape(tf.log(sparse_reduce_sum(sparse_exp(s), axis)), [-1])


def sparse_logsumexp(s, axis):
    if axis == 0:
        other_axis = 1
    elif axis == 1 or axis == -1:
        other_axis = 0

    raw_max = tf.sparse.reduce_max(s, axis=axis)
    mx = tf.stop_gradient(
        tf.where(
            tf.is_finite(raw_max), raw_max,
            tf.zeros_like(raw_max)))
    result = sparse_naive_logsumexp(sparse_matrix_dense_broadcasted_vector_add(s, -mx, axis), axis) + mx
    return result


def to_sparse(dense):
    # Find indices where the tensor is not zero
    idx = tf.where(tf.not_equal(dense, 0))
    # Make the sparse tensor
    # Use tf.shape(a_t, out_type=tf.int64) instead of a_t.get_shape()
    # if tensor shape is dynamic
    sparse = tf.SparseTensor(idx, tf.gather_nd(dense, idx), tf.shape(dense, out_type=tf.int64))
    return sparse


def to_dense(s):
    return tf.sparse.to_dense(s, validate_indices=False)
