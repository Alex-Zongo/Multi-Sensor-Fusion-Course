def q(x, mu, sigma):
    return np.exp(-(x - mu)**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)


def ratio(x):
    return f(x) / q(x, mu=0, sigma=1)


def rejection_sampling(n):
    samples = []
    for _ in range(n):
        x = np.random.normal(loc=0, scale=1)  # sample from q
        u = np.random.uniform(0, 1)
        if u < ratio(x):  # accept or reject samples based on the ratio
            samples.append(x)
    return np.array(samples)
