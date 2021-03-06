import numpy as np
import scipy as sp
import numexpr as ne
import fenics as fe
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from letkf_forecasting.advect import cython_space_deriv_4, cython_time_deriv_3
fe.set_log_level(30)


def _time_deriv_3(q, dt, u, dx, v, dy):
    k = space_deriv_4(q, u, dx, v, dy)
    k = space_deriv_4(q + dt/3*k, u, dx, v, dy)
    k = space_deriv_4(q + dt/2*k, u, dx, v, dy)
    qout = q + dt*k
    return qout


def _time_deriv_3_cython(q, dt, u, dx, v, dy):
    qout = np.empty_like(q)
    F_x = np.empty_like(v)
    F_y = np.empty_like(u)
    u_w = u[:, 0:2].clip(max=0)
    u_e = u[:, -2:].clip(min=0)
    v_n = v[-2:, :].clip(min=0)
    v_s = v[:2, :].clip(max=0)
    qint = np.empty_like(q)
    cython_time_deriv_3(q, qint, u, v, qout, F_x, F_y, dx, dy, dt,
                        u_w, u_e, v_n, v_s)
    return qout


time_deriv_3 = _time_deriv_3_cython


def _space_deriv_4_cython(q, u, dx, v, dy):
    qout = np.empty_like(q)
    fy = np.empty_like(v)
    fx = np.empty_like(u)
    u_w = u[:, 0:2].clip(max=0)  # noqa
    u_e = u[:, -2:].clip(min=0)  # noqa
    v_n = v[-2:, :].clip(min=0)  # noqa
    v_s = v[0:2, :].clip(max=0)  # noqa
    cython_space_deriv_4(q, u, v, qout, fx, fy, dx, dy,
                         u_w, u_e, v_n, v_s)
    return qout


def space_deriv_4(q, u, dx, v, dy):
    ne_eval = partial(ne.evaluate, optimization='aggressive')
    qout = np.zeros_like(q)
    F_x = np.zeros_like(u)
    F_y = np.zeros_like(v)

    # with numexpr
    u22 = u[:, 2:-2]  # noqa
    q21 = q[:, 2:-1]  # noqa
    q12 = q[:, 1:-2]  # noqa
    q3 = q[:, 3:]     # noqa
    qn3 = q[:, :-3]   # noqa
    F_x[:, 2:-2] = ne_eval('u22 / 12 * (7 * (q21 + q12) - (q3 + qn3))')

    v22 = v[2:-2, :]            # noqa
    q21 = q[2:-1, :]            # noqa
    q12 = q[1:-2, :]            # noqan
    q3 = q[3:, :]               # noqa
    qn3 = q[:-3, :]             # noqa
    F_y[2:-2, :] = ne_eval('v22 / 12 * (7 * (q21 + q12) - (q3 + qn3))')

    qo22 = qout[:, 2:-2]
    fx32 = F_x[:, 3:-2]         # noqa
    fx23 = F_x[:, 2:-3]         # noqa
    qout[:, 2:-2] = ne_eval('qo22 - (fx32 - fx23) / dx')

    qo22 = qout[2:-2, :]        # noqa
    fy32 = F_y[3:-2, :]         # noqa
    fy23 = F_y[2:-3, :]         # noqa
    qout[2:-2, :] = ne_eval('qo22 - (fy32 - fy23) / dy')

    # boundary calculation
    u_w = u[:, 0:2].clip(max=0)  # noqa
    u_e = u[:, -2:].clip(min=0)  # noqa

    qo02 = qout[:, 0:2]
    q13 = q[:, 1:3]
    q02 = q[:, 0:2]
    u13 = u[:, 1:3]             # noqa
    u02 = u[:, 0:2]             # noqa
    qout[:, 0:2] = ne_eval(
        'qo02 - ((u_w/dx)*(q13 - q02) + (q02/dx)*(u13 - u02))')

    qo2 = qout[:, -2:]
    q2 = q[:, -2:]
    q31 = q[:, -3:-1]
    u2 = u[:, -2:]              # noqa
    u31 = u[:, -3:-1]           # noqa
    qout[:, -2:] = ne_eval(
        'qo2 - ((u_e/dx)*(q2 - q31) + (q2/dx)*(u2 - u31))')

    v_n = v[-2:, :].clip(min=0)  # noqa
    v_s = v[0:2, :].clip(max=0)  # noqa

    qo02 = qout[0:2, :]         # noqa
    q13 = q[1:3, :]             # noqa
    q02 = q[0:2, :]             # noqa
    v13 = v[1:3, :]             # noqa
    v02 = v[0:2, :]             # noqa
    qout[0:2, :] = ne_eval(
        'qo02 - ((v_s/dx)*(q13 - q02) + (q02/dx)*(v13 - v02))')

    qo2 = qout[-2:, :]          # noqa
    q2 = q[-2:, :]              # noqa
    q31 = q[-3:-1, :]           # noqa
    v2 = v[-2:, :]              # noqa
    v31 = v[-3:-1, :]           # noqa
    qout[-2:, :] = ne_eval(
        'qo2 - ((v_n/dx)*(q2 - q31) + (q2/dx)*(v2 - v31))')
    return qout


def advect_5min(q, dt, U, dx, V, dy, T_steps):
    """Check back later"""
    for t in range(T_steps):
        q = time_deriv_3(q, dt, U, dx, V, dy)
    return q


def time_deriv_3_loop(CI_field, U, V, domain_shape, T_steps, dt, dx, dy):
    CI_field = CI_field.reshape(domain_shape)
    for t in range(T_steps):
        CI_field = time_deriv_3(np.ascontiguousarray(CI_field), dt,
                                np.ascontiguousarray(U), dx,
                                np.ascontiguousarray(V), dy)
    return CI_field.ravel()


def advect_5min_ensemble(
        ensemble, dt, dx, dy, T_steps, U_shape, V_shape, domain_shape,
        workers):

        """Check back later"""
        ens_size = ensemble.shape[1]
        U_size = U_shape[0]*U_shape[1]
        V_size = V_shape[0]*V_shape[1]
        wind_size = U_size + V_size

        CI_fields = ensemble[wind_size:].copy()
        CI_fields = CI_fields.T
        us = ensemble[:U_size].T.reshape(ens_size, U_shape[0], U_shape[1])
        vs = ensemble[U_size: V_size + U_size].T.reshape(
            ens_size, V_shape[0], V_shape[1])

        func = partial(time_deriv_3_loop, domain_shape=domain_shape,
                       T_steps=T_steps,
                       dt=dt, dx=dx, dy=dy)
        logging.debug('Running 5min ensemble advection')

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = executor.map(func,
                                   CI_fields, us, vs)
            temp = list(futures)
        temp = np.stack(temp, axis=1)
        ensemble[wind_size:] = temp
        logging.debug('Done with advection')
        return ensemble


def advect_5min_single(
        ensemble, dt, dx, dy, T_steps, U_shape, V_shape, domain_shape):
    U_size = U_shape[0]*U_shape[1]
    V_size = V_shape[0]*V_shape[1]
    wind_size = U_size + V_size

    CI_fields = ensemble[wind_size:].copy()
    CI_fields = CI_fields.reshape(domain_shape)
    U = ensemble[:U_size].reshape(U_shape[0], U_shape[1])
    V = ensemble[U_size: V_size + U_size].reshape(
        V_shape[0], V_shape[1])
    CI_fields = advect_5min(CI_fields, dt, U, dx, V, dy, T_steps)
    ensemble = np.concatenate([U.ravel(), V.ravel(), CI_fields.ravel()])
    return ensemble


def noise_fun(domain_shape):
    noise_init = np.zeros(domain_shape)
    noise_init[0:25, :] = 1
    noise_init[-25:, :] = 1
    noise_init[:, 0:25] = 1
    noise_init[:, -25:] = 1
    noise_init = sp.ndimage.gaussian_filter(noise_init, 12)
    return noise_init


def divergence(u, v, dx, dy):
    dudy, dudx = np.gradient(u, dy, dx)
    dvdy, dvdx = np.gradient(v, dy, dx)
    return dudx + dvdy


def remove_divergence(V, u, v, sigma):
    c_shape = u.shape
    V_div = divergence(u, v, 1, 1)
    ff = fe.Function(V)
    d2v_map = fe.dof_to_vertex_map(V)
    array_ff = V_div.ravel()
    array_ff = array_ff[d2v_map]
    ff.vector().set_local(array_ff)
    uu = fe.TrialFunction(V)
    vv = fe.TestFunction(V)
    a = fe.dot(fe.grad(uu), fe.grad(vv))*fe.dx
    L = ff*vv*fe.dx
    uu = fe.Function(V)
    fe.solve(a == L, uu)
    phi = uu.compute_vertex_values().reshape(c_shape)
    grad_phi = np.gradient(phi, 1, 1)
    u_corrected = u + grad_phi[1]
    v_corrected = v + grad_phi[0]
    u_corrected = sp.ndimage.filters.gaussian_filter(u_corrected, sigma=sigma)
    v_corrected = sp.ndimage.filters.gaussian_filter(v_corrected, sigma=sigma)
    return u_corrected, v_corrected


def remove_divergence_single(
        FunctionSpace, u, v, sigma):
    temp_u = u
    temp_u = .5*(temp_u[:, :-1] + temp_u[:, 1:])
    temp_v = v
    temp_v = .5*(temp_v[:-1, :] + temp_v[1:, :])
    temp_u, temp_v = remove_divergence(FunctionSpace,
                                       temp_u, temp_v, sigma)
    temp1 = np.pad(temp_u, ((0, 0), (0, 1)), mode='edge')
    temp2 = np.pad(temp_u, ((0, 0), (1, 0)), mode='edge')
    temp_u = .5*(temp1 + temp2)
    temp1 = np.pad(temp_v, ((0, 1), (0, 0)), mode='edge')
    temp2 = np.pad(temp_v, ((1, 0), (0, 0)), mode='edge')
    temp_v = .5*(temp1 + temp2)
    return temp_u, temp_v


def remove_divergence_ensemble(
        *, FunctionSpace, wind_ensemble, U_crop_shape, V_crop_shape, sigma):
    U_size = U_crop_shape[0]*U_crop_shape[1]
    V_size = V_crop_shape[0]*V_crop_shape[1]
    ens_size = wind_ensemble.shape[1]
    for ens_num in range(ens_size):
        logging.debug('Removing Divergence from ensemble # %s', ens_num)
        temp_u = wind_ensemble[:U_size, ens_num].reshape(U_crop_shape)
        temp_u = .5*(temp_u[:, :-1] + temp_u[:, 1:])
        temp_v = wind_ensemble[U_size:U_size + V_size,
                               ens_num].reshape(V_crop_shape)
        temp_v = .5*(temp_v[:-1, :] + temp_v[1:, :])
        temp_u, temp_v = remove_divergence(FunctionSpace,
                                           temp_u, temp_v, sigma)
        temp1 = np.pad(temp_u, ((0, 0), (0, 1)), mode='edge')
        temp2 = np.pad(temp_u, ((0, 0), (1, 0)), mode='edge')
        temp_u = .5*(temp1 + temp2)
        temp1 = np.pad(temp_v, ((0, 1), (0, 0)), mode='edge')
        temp2 = np.pad(temp_v, ((1, 0), (0, 0)), mode='edge')
        temp_v = .5*(temp1 + temp2)
        wind_ensemble[:U_size, ens_num] = temp_u.ravel()
        wind_ensemble[U_size:U_size + V_size, ens_num] = temp_v.ravel()
    return wind_ensemble
