import numpy as np
import tensorflow as tf
import cv2
import moviepy.editor as mvp
import itertools
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
# note: conditionally imports networkx

import sinkhorn


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
    sinkhorn_epsilon = 0.01 # TODO: 0.01 because currently sparse is imitating dense
    k = n # k = n means dense. TODO currently ignored, sparse imitating dense
    resample_targets = False
    # shrinking domain to avoid logsumexp trick.
    shrinkage = 0.1
    VIDEO_SIZE = 512

    np.random.seed(2)
    # first two coordinates are linearly transformed in an ad hoc way, rest simply multiplied by 2.
    start_np = np.random.uniform(size=(n, d)).astype(np.float32)
    start_np *= 0.05
    start_np[:, 0] += start_np[:, 1]
    start_np += 2
    target_np = np.random.normal(size=(n, d)).astype(np.float32)

    assert start_np.shape == target_np.shape == (n, d)

    target_np *= shrinkage
    start_np *= shrinkage

    print(np.mean(target_np[:, :4], axis=0), "\n", np.cov(target_np[:, :4].T))

    do_initial_matching = False
    do_rematching = False

    if do_initial_matching:
        initial_matching = optimalMatching(start_np, target_np)
        target_np = target_np[initial_matching]

    with tf.Session() as sess:
        pos = tf.Variable(start_np.astype(np.float32))
        if resample_targets:
            target = tf.random.normal((n, d), dtype=np.float32) * shrinkage
        else:
            target = tf.constant(target_np.astype(np.float32))

        sparse_indices_np  = [l for l in itertools.product(np.arange(n, dtype=np.int64), repeat=2)]
        sparse_indices = tf.constant(sparse_indices_np, dtype=np.int64)

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

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

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
                '''
                P_s_np, P_d_np = sess.run([sinkhorn.to_dense(P_s), P_d])
                print("P_d")
                print(P_d_np[:5, :5])
                print("P_s")
                print(P_s_np[:5, :5])
                C_s_np, C_d_np = sess.run([sinkhorn.to_dense(C_s), C_d])
                print("C_d")
                print(C_d_np[:5, :5])
                print("C_s")
                print(C_s_np[:5, :5])
                Q_s_np, Q_d_np = sess.run([sinkhorn.to_dense(Q_s), Q_d])
                print("Q_d")
                print(Q_d_np[:5, :5])
                print("Q_s")
                print(Q_s_np[:5, :5])

                f_s_np, f_d_np, OT_s_np, OT_d_np = sess.run([f_s, f_d, OT_s, OT_d])
                print("f_d")
                print(f_d_np)
                print("f_s")
                print(f_s_np)
                print("OT_d", OT_d_np)
                print("OT_s", OT_s_np)
                print("grad OT_d", sess.run(tf.gradients(OT_d, pos)))
                print("grad OT_s", sess.run(tf.gradients(OT_s, pos)))
                print("Q_d + C_d", (Q_d_np + C_d_np)[:5, :5])
                print("Q_s + C_s", (Q_s_np + C_s_np)[:5, :5])

                print("d C", np.linalg.norm(C_d_np - C_s_np))
                print("d P", np.linalg.norm(P_d_np - P_s_np))
                print("d Q", np.linalg.norm(Q_d_np - Q_s_np))
                print("d f", np.linalg.norm(f_d_np - f_s_np))
                print("d Q_d -C_d-f_d", np.linalg.norm(Q_d_np - (-C_d_np - f_d_np)))
                print("d Q_s -C_s-f_s", np.linalg.norm(Q_s_np - (-C_s_np - f_s_np)))
                '''

                if resample_targets:
                    _, next_pos_np, target_np = sess.run([train_step, pos, target])
                else:
                    _, next_pos_np = sess.run([train_step, pos])

                if do_rematching:
                    matching = optimalMatching(next_pos_np, target_np)
                    target_np_aligned = target_np[matching]
                else:
                    target_np_aligned = target_np

                draw_edges = do_initial_matching or do_rematching
                frame = sinkhorn.draw_edges(next_pos_np, target_np_aligned,
                                            VIDEO_SIZE, radius=4 * shrinkage, edges=draw_edges)
                video.write_frame(frame)

                print("iter:", indx, "transport:", sess.run(OT), "mean_length_of_matching:",
                    np.mean(np.linalg.norm(next_pos_np - target_np_aligned, axis=1)))
                print(np.mean(next_pos_np[:, :4], axis=0), "\n", np.cov(next_pos_np[:, :4].T))

main()
