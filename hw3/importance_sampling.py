def imp_pdf(x):
    return A * np.exp(-x)


def inverse_transform_sampling(a, b, n):
    u = np.random.uniform(a, b, n)
    return -np.log(1 - u/A)
