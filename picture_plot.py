import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def picsatlatent(images, pos):

    inum = len(images)

    def getImage(i, zoom=0.2):
        return OffsetImage(images[i][:,:,0], zoom=zoom)

    fig, ax = plt.subplots(subplot_kw = {'aspect' : 'equal'})

    for i in range(inum):
        ab = AnnotationBbox(getImage(i), (pos[i][0], pos[i][1]), frameon=False)
        ax.add_artist(ab)

    plt.xlim(-2,2)
    plt.ylim(-2,2)

    plt.savefig("latentpic.png", dpi = 1000)


'''
checkpoint = os.path.join(opts['work_dir'], 'checkpoints', 'trained-wae-100')

    net = wae.WAE(opts)
    net.saver.restore(net.sess, checkpoint)

    data = DataHandler(opts)
    data._load_mnist(opts)

    images = data.data[:1000]

    pos = net.enc_mean

    sess = tf.Session()
    with sess.as_default():
        pos = pos.eval()
'''