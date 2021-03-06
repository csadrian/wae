import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from interpolatingfuncs import lerp_gaussian
import wae
from models import encoder

import os
from datahandler import DataHandler

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import fideval # for restore_net


'''
n = 10
m = 10
zs = net.sample_pz(2 * n)
zs = np.reshape(zs, (n, 2, -1))

sample_gen = net.sess.run(net.decoded, feed_dict={net.sample_noise: grid, net.is_training: False})
'''

# test plot with randomly generated numbers
"""
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

images = x_train[:100]
images = np.expand_dims(images, axis=3)

pos = np.random.rand(100,2)
"""

# scatter images at specific positions
def scatterpics(images, pos):
    inum = len(images)
    
    def getImage(i, zoom=0.5):
        return OffsetImage(images[i][:,:,0], zoom=zoom)

    fig, ax = plt.subplots(subplot_kw = {'aspect' : 'equal'})

    for i in range(inum):
        ab = AnnotationBbox(getImage(i), (pos[i][0], pos[i][1]), frameon=False)
        ax.add_artist(ab)

    plt.xlim(-100,100)
    plt.ylim(-100,100)
    plt.savefig("latentpic.png", dpi = 1000)

# plot images in a grid
def plotImages(data, n_x, n_y, name, text=None):
    (height, width, channel) = data.shape[1:]
    height_inc = height + 1
    width_inc = width + 1
    n = len(data)
    if n > n_x*n_y: n = n_x * n_y

    if channel == 1:
        mode = "L"
        data = data[:,:,:,0]
        image_data = 50 * np.ones((height_inc * n_y + 1, width_inc * n_x - 1), dtype='uint8')
    else:
        mode = "RGB"
        image_data = 50 * np.ones((height_inc * n_y + 1, width_inc * n_x - 1, channel), dtype='uint8')
    for idx in range(n):
        x = idx % n_x
        y = idx // n_x
        sample = data[idx]
        image_data[height_inc*y:height_inc*y+height, width_inc*x:width_inc*x+width] = 255*sample.clip(0, 0.99999)
    img = Image.fromarray(image_data,mode=mode)
    fileName = name + ".png"
    # print("Creating file " + fileName)
    if text is not None:
        img.text(10, 10, text)
    img.save(fileName)

# interpolate between points
def interpolate(lows, highs):
    assert lows.shape == highs.shape
    n, d = lows.shape
    m = 5
    # (interpolate, 2, latent_dim)
    grid = np.zeros((n, m, d))
    for i, (low, high) in enumerate(zip(lows, highs)):
        for j, val in enumerate(np.linspace(0, 1, m)):
            grid[i, j, :] = lerp_gaussian(val, low, high)
    grid = np.reshape(grid, (n*m,d))   
    return grid


# My attemt at integrating it to the code

def createimgs(opts):
    net = fideval.restore_net(opts)

    n = 50
    k = 10
    m = 5

    NUM_POINTS = 10000
    BATCH_SIZE = 100

    data = DataHandler(opts)
    images = data.data[:n]

    enc_pics = net.sess.run(net.encoded,
                feed_dict={
                    #net.sample_noise: np.random.normal(size=(5, opts['zdim'])),
                    net.sample_points: images,
                    net.is_training: False
                })

    #pos = net.sample_pz(n)
    #pos = pos[:,:2]

    #pca = PCA(n_components=2)
    #pos = pca.fit_transform(pos)

    pos = enc_pics

    tsne = TSNE(n_components=2)
    pos = tsne.fit_transform(pos)

    print(pos)

    zs = net.sample_pz(2 * k)
    zs = np.reshape(zs, (k, 2, -1))
    lows = zs[:,0,:]
    highs = zs[:,1,:]
    grid = interpolate(lows, highs)
    
    for img_index in range(NUM_POINTS//BATCH_SIZE):
        gen_pics = net.sess.run(net.decoded,
                feed_dict={
                    #net.sample_noise: np.random.normal(size=(5, opts['zdim'])),
                    net.sample_noise: grid,
                    net.is_training: False
                })

    plotImages(gen_pics, m, k, 'gridpics')

    scatterpics(images, pos)

"""
n = 2
m = 5

zs = np.random.rand(2, 2, 2)
lows = zs[:,0,:]
highs = zs[:,1,:]
print("original", zs)
res = interpolate(lows, highs)
print("interpolated", res.shape, res)
"""
