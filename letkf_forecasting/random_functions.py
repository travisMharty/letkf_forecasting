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


def perturb_irradiance(ensemble, domain_shape, edge_weight, pert_mean,
                       pert_sigma, rf_approx_var, rf_eig, rf_vectors):
    ens_size = ensemble.shape[1]
    average = ensemble.mean(axis=1)
    average = average.reshape(domain_shape)
    target = ski_filters.sobel(average)
    target = target/target.max()
    target[target < 0.1] = 0
    target = sp.ndimage.gaussian_filter(target, sigma=4)
    target = target/target.max()
    cloud_target = 1 - average
    cloud_target = (cloud_target/cloud_target.max()).clip(min=0,
                                                          max=1)
    target = np.maximum(cloud_target, target*edge_weight)
    target = target/target.max()
    target = sp.ndimage.gaussian_filter(target, sigma=5)
    target = target.ravel()
    sample = np.random.randn(rf_eig.size, ens_size)
    sample = rf_vectors.dot(np.sqrt(rf_eig[:, None])*sample)
    target_mean = target.mean()
    target_var = (target**2).mean()
    cor_mean = pert_mean/target_mean
    cor_sd = pert_sigma/np.sqrt(rf_approx_var*target_var)
    ensemble = (
        ensemble +
        (cor_sd*sample + cor_mean)*target[:, None])
    return ensemble


def logistic(array, L, k, x0):
    return L/(1 + np.exp(-k*(array - x0)))


def perturb_irradiance_new(ensemble, domain_shape, edge_weight, pert_mean,
                           pert_sigma, rf_approx_var, rf_eig, rf_vectors):
    L = 1
    k = 20
    x0 = 0.2
    ens_size = ensemble.shape[1]
    average = ensemble.mean(axis=1)
    average = average.reshape(domain_shape)
    cloud_target = 1 - average
    cloud_target = logistic(cloud_target, L=L, k=k, x0=x0)
    cloud_target = sp.ndimage.maximum_filter(cloud_target, size=9)
    cloud_target = sp.ndimage.gaussian_filter(cloud_target, sigma=5)
    cloud_target = cloud_target/cloud_target.max()
    cloud_target = cloud_target.clip(min=0, max=1)
    cloud_target = cloud_target.ravel()

    sample = np.random.randn(rf_eig.size, ens_size)
    sample = rf_vectors.dot(np.sqrt(rf_eig[:, None])*sample)
    target_mean = cloud_target.mean()
    target_var = (cloud_target**2).mean()
    cor_mean = pert_mean/target_mean
    cor_sd = pert_sigma/np.sqrt(rf_approx_var*target_var)
    # cor_sd = pert_sigma/np.sqrt(rf_approx_var)
    ensemble = (
        ensemble +
        (cor_sd*sample + cor_mean)*cloud_target[:, None])
    ensemble = ensemble.clip(min=ensemble.min(), max=1)
    return ensemble
