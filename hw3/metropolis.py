def q(x, mu, sigma):
    return np.exp(-(x - mu)**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)


def sample_q(x, sigma):
    return np.random.normal(loc=x, scale=sigma)


def ratio(x, x_new, sigma):
    return min(1, f(x_new) * q(x, x_new, sigma) / (f(x) * q(x_new, x, sigma)))


def metropolis_hastings(n, sigma):
    samples = []
    x = 0
    for _ in range(n):
        x_new = sample_q(x, sigma)
        u = np.random.uniform(0, 1)
        if u < ratio(x, x_new, sigma):
            x = x_new
        samples.append(x)
    return np.array(samples)
