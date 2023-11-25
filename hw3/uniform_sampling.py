def uni_pdf(x):
    return 1/(b - a)


def monte_carlo_integration(f, a, b, n, dist, pdf):
    x = dist(a, b, n)
    h = (b - a)
    y = f(x)/pdf(x)

    return np.mean(y)
