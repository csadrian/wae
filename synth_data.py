import numpy as np

w = 28

intensity = 0.1



def checkers_sampler(xn, yn):
    bits = intensity + (1 - 2 * intensity) * np.random.randint(2, size=(xn, yn))
    img = np.kron(bits, np.ones((w // xn, w // yn)))
    # TODO hardwired 1 / 3.
    img += np.random.normal(scale=intensity / 3, size=(img.shape))
    return img


def checkers_sampler_n(n, xn, yn):
    w = 28
    ds = np.zeros((n, w, w))
    for i in range(n):
        ds[i, :, :] = checkers_sampler(xn, yn)
    return ds


def loglikelihood_estimator(d):
    xw = w // xn
    yw = w // yn
    blocks = d.reshape((xw, xn, yw, yn))
    bit_estimates = blocks.mean(axis=(1, 3))
    bits = (bit_estimates > 0.5).astype(np.int).astype(np.float)
    intended_img = np.kron(bits, np.ones((w // xn, w // yn)))
    return -np.sum(np.square(d - intended_img))


def main():
    xn = 4 # number of blocks horizontally
    yn = 4 # number of blocks vertically

    assert w % xn == w % yn == 0

    d = checkers_sampler(xn, yn)

    print(loglikelihood_estimator(d))
    print(loglikelihood_estimator(d + 0.1))
    print(loglikelihood_estimator(d / 2))


if __name__ == "__main__":
    main()
