import numpy as np

w = 28

intensity = 0.2


def checkers_sampler(xn, yn):
    low = np.random.uniform(low=0, high=intensity)
    high = 1 - np.random.uniform(low=0, high=intensity)
    bits = np.random.randint(2, size=(xn, yn))
    vals = low + (high - low) * bits
    img = np.kron(vals, np.ones((w // xn, w // yn)))
    # img = np.clip(img, 0.0, 1.0)
    return img


def checkers_sampler_n(n, xn, yn):
    ds = np.zeros((n, w, w))
    for i in range(n):
        ds[i, :, :] = checkers_sampler(xn, yn)
    return ds


def get_nearest_params(x):
    x = x.reshape((x.shape[0], -1))
    lows = np.average(x, weights=(x < 0.5) + 1e-9, axis=-1)
    highs = np.average(x, weights=(x > 0.5) + 1e-9, axis=-1)

    # we give it the benefit of doubt if there are no pixels that could inform us.
    # note that this can legitimately happen with a 1 / 2 ** (xn * yn) probability.
    lows[np.all(x >= 0.5, axis=-1)] = intensity / 2
    highs[np.all(x <= 0.5, axis=-1)] = 1 - intensity / 2

    lows /= intensity
    highs = (1 - highs) / intensity
    merged = np.column_stack((lows, highs))
    merged = np.clip(merged, 0.0, 1.0)
    return merged


def loglikelihood_estimator(d, xn, yn):
    assert False, "this does not consider 2d intensity"
    xw = w // xn
    yw = w // yn
    blocks = d.reshape((xn, xw, yn, yw))
    bit_estimates = blocks.mean(axis=(1, 3))
    bits = (bit_estimates > 0.5).astype(np.int).astype(np.float)
    intended_img = np.kron(bits, np.ones((xw, yw)))
    return -np.sum(np.square(d - intended_img))


def main():
    xn = 4 # number of blocks horizontally
    yn = 4 # number of blocks vertically

    assert w % xn == w % yn == 0

    d = checkers_sampler(xn, yn)

    # print(loglikelihood_estimator(d, xn, yn))
    # print(loglikelihood_estimator(d + 0.1, xn, yn))
    # print(loglikelihood_estimator(d / 2, xn, yn))

    ds = checkers_sampler_n(1000, xn, yn)
    ps = get_nearest_params(ds)
    bins = (ps * 10).astype(np.int)
    for v in (bins[:, 0] + 10 * bins[:, 1]):
        print(v)


if __name__ == "__main__":
    main()
