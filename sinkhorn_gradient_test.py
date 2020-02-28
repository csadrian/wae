import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sinkhorn


def main():
    n, d = 3, 2
    
    latents = tf.placeholder(
            tf.float32, (n, d), name='real_points_ph')
    targets = tf.placeholder(
            tf.float32, (n, d), name='noise_ph')
    OT, P_temp, P, f, g, C = sinkhorn.SinkhornLoss(latents, targets, epsilon=0.01, niter=10)

    OTgrad = tf.gradients(OT, latents)[0]
    with tf.Session() as sess:
        latents_np_start = np.random.normal(size=(n, d))
        targets_np = np.random.normal(size=(n, d))
        N = 10
        pairs = []
        ots = []
        otgrads = []
        alphas = np.linspace(0, 1, N)
        if False:
            plt.scatter(latents_np_start[:, 0], latents_np_start[:, 1], c="blue")
            plt.scatter(targets_np[:, 0], targets_np[:, 1], c="red")
            plt.show()
        just_one = True
        for i in range(N):
            alpha = alphas[i]
            latents_np = latents_np_start.copy()
            # large alpha means target latents[0] is close to targets[0]
            if just_one:
                latents_np[0] = alpha * latents_np_start[0] + (1 - alpha) * targets_np[0]
                # latents_np[1:] = targets_np[1:]
            else:
                latents_np = alpha * latents_np_start + (1 - alpha) * targets_np

            feed_dict = {
                latents: latents_np,
                targets: targets_np
            }
            p = sess.run(tf.exp(P), feed_dict = feed_dict)
            pair = np.argmax(p[0])
            pairs.append(pair)
            ot = sess.run(OT, feed_dict=feed_dict)
            ots.append(ot)
            otgrad_np = sess.run(OTgrad, feed_dict=feed_dict)[0]
            # print(alpha, ot, otgrad_np)
            otgrads.append(otgrad_np)

            fig, ax = plt.subplots()
            plt.scatter(latents_np[:, 0], latents_np[:, 1], c="blue")
            plt.scatter(targets_np[:, 0], targets_np[:, 1], c="red")
            for j in range(n):
                ax.plot([latents_np[0, 0], targets_np[j, 0]], [latents_np[0, 1], targets_np[j, 1]])
                ax.annotate(p[0, j], targets_np[j])
            ax.quiver(latents_np[0, 0], latents_np[0, 1],
                      -otgrad_np[0], -otgrad_np[1],
                      angles = 'xy', scale_units = 'xy', scale = 1, width = 0.001)
            plt.title("Couplings of first point.\niter=%d" % i)
            plt.savefig("Coupling_iter=%d" % i)
            plt.close()
            
        print("Pairs: ", pairs)
        ots = np.array(ots)
        otgrads = np.array(otgrads)
        fig, ax1 = plt.subplots()
        if just_one:
            plt.title("z[0] interpolates between targets[0] and fully random position,\nthe rest of the points are fixed random.\nn=%d, d=%d" % (n, d))
        else:
            plt.title("z interpolates between targets and fully random position.\nn=%d, d=%d" % (n, d))
        ax1.set_xlabel("alpha, smaller means z closer to target")
        ax1.set_ylabel("dOT/dz[0] gradient magnitude", color="blue")
        ax1.plot(alphas, np.linalg.norm(otgrads, axis=1), c="blue")
        ax2 = ax1.twinx()
        ax2.tick_params(axis='y', labelcolor="red")
        ax2.plot(alphas, ots, c="red")
        ax2.set_ylabel("OT", color="red")

        # plt.plot(alphas, otgrads[:, 0], c="blue") ; plt.plot(alphas, otgrads[:, 1], c="green")
        plt.savefig("OT_grads.png")
        #plt.show()


main()
