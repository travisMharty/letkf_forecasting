import numpy as np


def get_cov(n, L, dx):
    C = np.zeros([n, n])
    for ii in np.arange(n):
        for jj in np.arange(ii, n):
            dist = abs(ii - jj)
            # dist = np.min([abs(ii - jj), np.mod(-abs(ii - jj), n)])
            C[ii, jj] = np.exp(-(dist*dx)**2/(2*L**2))
    return (C + C.T) - np.diag(np.diag(C))


def get_random_function_1d(x, L):
    dx = x[1] - x[0]
    n = x.size
    C = get_cov(n, L, dx)

    e, v = np.linalg.eigh(C)
    e = e[::-1]
    v = v[:, ::-1]

    go = 1
    counter = 1
    tol = 0.05
    while go == 1:
        if (e[:counter]**2).sum() > (1 - tol)*(e**2).sum():
            go = 0
        else:
            counter += 1

    sqrt_C_hat = v[:, :counter].dot(np.diag(np.sqrt(e[:counter])))
    y = sqrt_C_hat.dot(np.random.randn(counter, 1))
    return y


def approx(e, tol):
    eig_num = 1
    go = 1
    while go == 1:
        if (e[:eig_num]**2).sum() > (1 - tol)*(e**2).sum():
            go = 0
        else:
            eig_num += 1
    return eig_num


def get_approx_eig(C, tol):
    e, v = np.linalg.eigh(C)
    e = e[::-1]
    v = v[:, ::-1]
    eig_num = approx(e, tol)
    return e[:eig_num], v[:, :eig_num]


def get_random_function_2d(x, y, Lx, Ly, tol):
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    nx = x.size
    ny = y.size
    Cx = get_cov(nx, Lx, dx)
    Cy = get_cov(ny, Ly, dy)
    ex, vx = get_approx_eig(Cx, tol)
    ey, vy = get_approx_eig(Cy, tol)
    e = np.kron(ey, ex)
    v = np.kron(vy, vx)

    # even more reduction
    sorted_indices = np.argsort(e)
    e = e[sorted_indices]
    v = v[:, sorted_indices]
    eig_num = get_approx_eig(e)
    e = e[:eig_num]
    v = v[:, :eig_num]
    z = v.dot((np.sqrt(e)*np.random.randn(eig_num))[:, None])
    return z.reshape(ny, nx)


def eig_2d_covariance(x, y, Lx, Ly, tol):
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    nx = x.size
    ny = y.size
    Cx = get_cov(nx, Lx, dx)
    Cy = get_cov(ny, Ly, dy)
    ex, vx = get_approx_eig(Cx, tol)
    ey, vy = get_approx_eig(Cy, tol)
    e = np.kron(ey, ex)
    v = np.kron(vy, vx)

    # even more reduction
    sorted_indices = np.argsort(e)
    sorted_indices = sorted_indices[::-1]
    e = e[sorted_indices]
    v = v[:, sorted_indices]
    eig_num = approx(e, tol)
    e = e[:eig_num]
    v = v[:, :eig_num]
    return e, v
