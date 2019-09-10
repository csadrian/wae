import numpy as np
import tensorflow as tf
import cv2
import moviepy.editor as mvp
import itertools
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
# note: conditionally imports networkx

import sinkhorn
import sparsifiers


def pairwiseSquaredDistances(clients, servers):
    cL2S = np.sum(clients ** 2, axis=-1)
    sL2S = np.sum(servers ** 2, axis=-1)
    cL2SM = np.tile(cL2S, (len(servers), 1))
    sL2SM = np.tile(sL2S, (len(clients), 1))
    squaredDistances = cL2SM + sL2SM.T - 2.0 * servers.dot(clients.T)
    return np.clip(squaredDistances.T, a_min=0, a_max=None)


# Wasserstein_1, that is, L2 rather than squared L2.
def optimalMatching(latentPositions, natPositions):
    import networkx as nx

    distances = np.sqrt(pairwiseSquaredDistances(latentPositions, natPositions))
    n = distances.shape[0]
    bipartite = nx.Graph()
    bipartite.add_nodes_from(range(n), bipartite=0)
    bipartite.add_nodes_from(range(n, 2*n), bipartite=1)
    for i in range(n):
        for j in range(n):
            bipartite.add_edge(i, n+j, weight=-distances[i, j], near=False)
    matching = nx.algorithms.matching.max_weight_matching(bipartite, maxcardinality=True, weight='weight')
    m2 = [None for _ in range(n)]
    for a, b in matching:
        if a >= n:
            b, a = a, b
        latent_index = a
        nat_index = b - n
        m2[latent_index] = nat_index
    return m2


def grid(a, b):
    x = np.linspace(-1, 1, a)
    y = np.linspace(-1, 1, b)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    return XX.astype(np.float32)


def main():
    use_sparse = True
    n = 200
    d = 64
    step_count = 100
    sinkhorn_iters = 10
    sinkhorn_epsilon = 0.1 if use_sparse else 0.01

    # sparsifier_kind is overridden by use_sparse
    sparsifier_kind = "topk"
    # dense: complete interaction graph. only works with use_sparse=False
    # topk: top k target point chosen for each moved point
    # twoway-topk: top k target point chosen for each moved point, then other way round, union is taken.
    # random: Renyi random interaction, resampled in every iteration
    # random-without-resample: Renyi random interaction, fixed at startup
    assert sparsifier_kind in ("dense", "topk", "twoway-topk", "random", "random-without-resample")

    k = 50 # for "random" kinds, k * n is the number of sampled interactions
    resample_targets = False
    VIDEO_SIZE = 512

    np.random.seed(2)
    # first two coordinates are linearly transformed in an ad hoc way, rest simply multiplied by 2.
    start_np = np.random.uniform(size=(n, d)).astype(np.float32)
    start_np *= 0.05
    start_np[:, 0] += start_np[:, 1]
    start_np += 2
    target_np = np.random.normal(size=(n, d)).astype(np.float32)

    assert start_np.shape == target_np.shape == (n, d)

    print(np.mean(target_np[:, :4], axis=0), "\n", np.cov(target_np[:, :4].T))

    do_initial_matching = False
    do_rematching = False

    if do_initial_matching:
        initial_matching = optimalMatching(start_np, target_np)
        target_np = target_np[initial_matching]

    with tf.Session() as sess:
        pos = tf.Variable(start_np.astype(np.float32))
        if resample_targets:
            target = tf.random.normal((n, d), dtype=np.float32)
        else:
            target = tf.constant(target_np.astype(np.float32))

        sparse_indices = tf.placeholder(np.int64, (None, 2))

        # Q are used for debugging.
        print("building sinkhorn ops graph")
        OT_s, Q_s, P_s, f_s, g_s, C_s = sinkhorn.SparseSinkhornLoss(target, pos, sparse_indices, epsilon=sinkhorn_epsilon, niter=sinkhorn_iters)
        OT_d, Q_d, P_d, f_d, g_d, C_d = sinkhorn.SinkhornLoss(target, pos, epsilon=sinkhorn_epsilon, niter=sinkhorn_iters)

        if use_sparse:
            OT, P, f, g, C = OT_s, P_s, f_s, g_s, C_s
        else:
            OT, P, f, g, C = OT_d, P_d, f_d, g_d, C_d

        # randomly throwing away elements of C, no importance sampling:
        # OT, P, f, g, C = sinkhorn.EmulatedSparseSinkhornLoss(pos, target, epsilon=0.01, niter=10)
        # adjusted with autocorrelation terms:
        # OT, P, f, g, C = sinkhorn.SinkhornDivergence(pos, target, epsilon=0.01, niter=10)

        if not use_sparse or sparsifier_kind == "dense":
            sparsifier = None
        elif sparsifier_kind == "topk":
            sparsifier = sparsifiers.TfTopkSparsifier(pos, target, k, sess, batch_size=min(n, 1000))
        elif sparsifier_kind == "twoway-topk":
            sparsifier = sparsifiers.TfTwoWayTopkSparsifier(pos, target, k, sess, batch_size=min(n, 1000))
        elif sparsifier_kind == "random":
            sparsifier = sparsifiers.RandomSparsifier(n, n, k * n, resample=True)
        elif sparsifier_kind == "random-without-resample":
            sparsifier = sparsifiers.RandomSparsifier(n, n, k * n, resample=False)
        else:
            assert False, "unknown sparsifier kind"

        optimizer = tf.train.AdamOptimizer(learning_rate=0.1)

        print("building grad op")
        train_step = optimizer.minimize(OT, var_list=pos)
        print("building grad op done")

        # did not help anyway
        clip_gradients = False
        if clip_gradients:
            grad_vars = optimizer.compute_gradients(OT)
            clipped_gvs = [(tf.clip_by_value(grad, -100, +100), var) for grad, var in grad_vars]
            train_step = optimizer.apply_gradients(clipped_gvs)

        sess.run(tf.global_variables_initializer())

        with FFMPEG_VideoWriter('out.mp4', (VIDEO_SIZE, VIDEO_SIZE), 30.0) as video:
            for indx in range(step_count):
                if use_sparse and sparsifier_kind != "dense":
                    sparse_indices_np = sparsifier.indices()
                else:
                    sparse_indices_np = np.zeros((0, 2)).astype(np.int64)

                # TODO !!! needed because sparsifier constructor takes (pos, target)
                # and SparseSinkhornLoss takes (target, pos).
                sparse_indices_np = sparse_indices_np[:, ::-1]

                if resample_targets:
                    _, next_pos_np, target_np = sess.run([train_step, pos, target],
                        feed_dict={sparse_indices: sparse_indices_np})
                else:
                    _, next_pos_np = sess.run([train_step, pos],
                        feed_dict={sparse_indices: sparse_indices_np})

                if do_rematching:
                    matching = optimalMatching(next_pos_np, target_np)
                    target_np_aligned = target_np[matching]
                else:
                    target_np_aligned = target_np

                draw_edges = do_initial_matching or do_rematching
                frame = sinkhorn.draw_edges(next_pos_np, target_np_aligned,
                                            VIDEO_SIZE, radius=4, edges=draw_edges)
                video.write_frame(frame)

                OT_np = sess.run(OT,
                        feed_dict={sparse_indices: sparse_indices_np})

                print("iter:", indx, "transport:", OT_np, "mean_length_of_matching:",
                    np.mean(np.linalg.norm(next_pos_np - target_np_aligned, axis=1)))
                print(np.mean(next_pos_np[:, :4], axis=0), "\n", np.cov(next_pos_np[:, :4].T))

main()
