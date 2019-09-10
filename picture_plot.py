import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from scipy.stats import norm
import wae




'''
n = 10
m = 10
zs = net.sample_pz(2 * n)
zs = np.reshape(zs, (n, 2, -1))

sample_gen = net.sess.run(net.decoded, feed_dict={net.sample_noise: grid, net.is_training: False})
'''

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

images = x_train[:100]
images = np.expand_dims(images, axis=3)

pos = np.random.rand(100,2)

def picsatpos(images, pos):

    inum = len(images)
    
    def getImage(i, zoom=0.5):
        return OffsetImage(images[i][:,:,0], zoom=zoom)
    '''
    def getImage(i, zoom=0.5):
        return OffsetImage(images[i], zoom=zoom)
    '''
    fig, ax = plt.subplots(subplot_kw = {'aspect' : 'equal'})

    for i in range(inum):
        ab = AnnotationBbox(getImage(i), (pos[i][0], pos[i][1]), frameon=False)
        ax.add_artist(ab)

    #plt.xlim(-2,2)
    #plt.ylim(-2,2)

    plt.savefig("latentpic.png", dpi = 1000)


picsatpos(images, pos)

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

plotImages(images, 10, 10, 'latentgrid')

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

def lerp(val, low, high):
    """Linear interpolation"""
    return low + (high - low) * val

def lerp_gaussian(val, low, high):
    """Linear interpolation with gaussian CDF"""
    low_gau = norm.cdf(low)
    high_gau = norm.cdf(high)
    lerped_gau = lerp(val, low_gau, high_gau)
    return norm.ppf(lerped_gau)

def slerp(val, low, high):
    """Spherical interpolation. val has a range of 0 to 1."""
    if val <= 0:
        return low
    elif val >= 1:
        return high
    elif np.allclose(low, high):
        return low
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high

def slerp_gaussian(val, low, high):
    """Spherical interpolation with gaussian CDF (generally not useful)"""
    offset = norm.cdf(np.zeros_like(low))  # offset is just [0.5, 0.5, ...]
    low_gau_shifted = norm.cdf(low) - offset
    high_gau_shifted = norm.cdf(high) - offset
    circle_lerped_gau = slerp(val, low_gau_shifted, high_gau_shifted)
    epsilon = 0.001
    clipped_sum = np.clip(circle_lerped_gau + offset, epsilon, 1.0 - epsilon)
    result = norm.ppf(clipped_sum)
    return result

def get_interpfn(spherical, gaussian):
    """Returns an interpolation function"""
    if spherical and gaussian:
        return slerp_gaussian
    elif spherical:
        return slerp
    elif gaussian:
        return lerp_gaussian
    else:
        return lerp

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

n = 2
m = 5

zs = np.random.rand(2, 2, 2)
lows = zs[:,0,:]
highs = zs[:,1,:]
print("original", zs)
res = interpolate(lows, highs)
print("interpolated", res.shape, res)
