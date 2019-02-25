import numpy as np
import scipy as sp
from skimage import filters as ski_filters


def get_cov(n, L, dx):
    C = np.zeros([n, n])
    for ii in np.arange(n):
        for jj in np.arange(ii, n):
            dist = abs(ii - jj)
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

    sorted_indices = np.argsort(e)
    sorted_indices = sorted_indices[::-1]
    e = e[sorted_indices]
    v = v[:, sorted_indices]
    eig_num = approx(e, tol)
    e = e[:eig_num]
    v = v[:, :eig_num]
    return e, v


def perturb_irradiance_old(ensemble, domain_shape, edge_weight, pert_mean,
                       pert_sigma, rf_approx_var, rf_eig, rf_vectors):
    ens_size = ensemble.shape[1]
    average = ensemble.mean(axis=1)
    average = average.reshape(domain_shape)
    target = ski_filters.sobel(average)
    target = target/target.max()
    target[target < 0.1] = 0
    target = sp.ndimage.gaussian_filter(target, sigma=4)
    target = target/target.max()
    cloud_target = average
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


def perturb_irradiance(ensemble, domain_shape, edge_weight, pert_mean,
                       pert_sigma, rf_approx_var, rf_eig, rf_vectors):
    L = 1
    k = 20
    x0 = 0.2
    ens_size = ensemble.shape[1]
    average = ensemble.mean(axis=1)
    average = average.reshape(domain_shape)
    cloud_target = average.copy()
    cloud_target = logistic(cloud_target, L=L, k=k, x0=x0)
    size = 19
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    x = x - x.mean()
    y = y - y.mean()
    rho = np.sqrt(x**2 + y**2)
    rho_min = rho[:, 0].min()
    footprint = rho < rho_min
    cloud_target = cloud_target/cloud_target.max()
    cloud_target = cloud_target.clip(min=0, max=1)
    cloud_target = sp.ndimage.maximum_filter(cloud_target,
                                             footprint=footprint)
    cloud_target = sp.ndimage.gaussian_filter(cloud_target, sigma=5)
    cloud_target = cloud_target.ravel()

    sample = np.random.randn(rf_eig.size, ens_size)
    sample = rf_vectors.dot(np.sqrt(rf_eig[:, None])*sample)
    target_mean = cloud_target.mean()
    cor_mean = pert_mean/target_mean
    cor_sd = pert_sigma/np.sqrt(rf_approx_var)
    max_ci = np.min([1, ensemble.max()*1.05])
    ensemble = (
        ensemble +
        (cor_sd*sample + cor_mean)*cloud_target[:, None])
    ensemble = ensemble.clip(min=0, max=max_ci)
    return ensemble


def generate_random_winds(
         rf_vectors, rf_eig, sigma, Ne, shape, dx):
    stream = np.random.randn(rf_eig.size, Ne)
    stream = rf_vectors.dot(
        np.sqrt(rf_eig[:, None])*stream)
    stream = stream.reshape(shape[0], shape[1], Ne)
    grad = np.array(np.gradient(stream, dx, axis=(0, 1),))
    U = grad[0]*sigma
    V = -grad[1]*sigma
    temp1 = np.pad(U, ((0, 0), (0, 1), (0, 0)), mode='edge')
    temp2 = np.pad(U, ((0, 0), (1, 0), (0, 0)), mode='edge')
    U = .5*(temp1 + temp2)
    temp1 = np.pad(V, ((0, 1), (0, 0), (0, 0)), mode='edge')
    temp2 = np.pad(V, ((1, 0), (0, 0), (0, 0)), mode='edge')
    V = .5*(temp1 + temp2)
    return U, V


def perturb_winds(ensemble, sys_vars, pert_params):
    ens_size = ensemble.shape[1]
    dx = sys_vars.dx/1000       # want dx in km not m
    U, V = generate_random_winds(sys_vars.rf_vectors_wind,
                                 sys_vars.rf_eig_wind,
                                 pert_params['pert_sigma_wind'],
                                 ens_size,
                                 sys_vars.ci_crop_shape,
                                 dx)
    ensemble[:sys_vars.U_crop_size] += U.reshape(
        sys_vars.U_crop_size, ens_size)
    ensemble[sys_vars.U_crop_size: sys_vars.wind_size] += V.reshape(
        sys_vars.V_crop_size, ens_size)
    return ensemble
