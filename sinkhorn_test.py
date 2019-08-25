import numpy as np
import tensorflow as tf
import cv2
import moviepy.editor as mvp
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


def main():
    n = 200
    d = 20
    VIDEO_SIZE = 512

    # first two coordinates are linearly transformed in an ad hoc way, rest simply multiplied by 2.
    start_np = np.random.normal(size=(n, d)).astype(np.float32)
    start_np *= 2
    start_np[:, 0] += start_np[:, 1]
    start_np += 2

    target_np = np.random.normal(size=(n, d)).astype(np.float32)
    print(np.mean(target_np[:, :4], axis=0), "\n", np.cov(target_np[:, :4].T))

    do_initial_matching = False
    do_rematching = False

    if do_initial_matching:
        initial_matching = optimalMatching(start_np, target_np)
        target_np = target_np[initial_matching]

    with tf.Session() as sess:
        pos = tf.Variable(start_np.astype(np.float32))
        target = tf.constant(target_np.astype(np.float32))

        C = sinkhorn.pdist(pos, target) / (0.01) ** 2
        P, f, g = sinkhorn.Sinkhorn(C, n=n, m=n, f=None, epsilon=0.01, niter=10)

        # g = tf.matmul(P, target) * n - pos
        # next_pos = pos + 0.1 * g
        OT = tf.reduce_mean(P * C) * n

        optimizer = tf.train.AdamOptimizer(learning_rate=0.1)

        train_step = optimizer.minimize(OT, var_list=pos)

        # did not help anyway
        clip_gradients = False
        if clip_gradients:
            grad_vars = optimizer.compute_gradients(OT)
            clipped_gvs = [(tf.clip_by_value(grad, -100, +100), var) for grad, var in grad_vars]
            train_step = optimizer.apply_gradients(clipped_gvs)

        sess.run(tf.global_variables_initializer())


        with FFMPEG_VideoWriter('out.mp4', (VIDEO_SIZE, VIDEO_SIZE), 30.0) as video:
            for indx in range(300):
                sess.run(train_step)
                next_pos_np = sess.run(pos)
                # frame = sinkhorn.draw_points(next_pos_np, VIDEO_SIZE)
                # frame = sinkhorn.draw_edges(next_pos_np[next_pos_np[:, 0].argsort()], target_np[target_np[:, 0].argsort()], VIDEO_SIZE)

                if do_rematching:
                    matching = optimalMatching(next_pos_np, target_np)
                    target_np_aligned = target_np[matching]
                else:
                    target_np_aligned = target_np

                draw_edges = do_initial_matching or do_rematching
                frame = sinkhorn.draw_edges(next_pos_np, target_np_aligned, VIDEO_SIZE, edges=draw_edges)
                video.write_frame(frame)

                print("iter:", indx, "transport:", sess.run(OT), "mean_length_of_matching:",
                    np.mean(np.linalg.norm(next_pos_np - target_np_aligned, axis=1)))
                print(np.mean(next_pos_np[:, :4], axis=0), "\n", np.cov(next_pos_np[:, :4].T))

main()
