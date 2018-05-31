import cython


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.cdivision_warnings(True)
cpdef void cython_space_deriv_4(
    double [:, ::1] q, double [:, ::1] u, double [:, ::1] v,
    double [:, ::1] qout, double [:, ::1] F_x, double [:, ::1] F_y,
    int dx, int dy, double [:, ::1] u_w, double [:, ::1] u_e,
    double [:, ::1] v_n, double [:, ::1] v_s) nogil:
    cdef Py_ssize_t u_shape_0 = u.shape[0]
    cdef Py_ssize_t u_shape_1 = u.shape[1]
    cdef Py_ssize_t v_shape_0 = v.shape[0]
    cdef Py_ssize_t v_shape_1 = v.shape[1]
    cdef Py_ssize_t q_shape_0 = q.shape[0]
    cdef Py_ssize_t q_shape_1 = q.shape[1]

    cdef Py_ssize_t j
    cdef Py_ssize_t i
    for i in range(u_shape_0):
        for j in range(2, u_shape_1 - 2):
            F_x[i, j] = u[i, j] / 12 * (7 * (
                q[i, j] + q[i, j-1]) - (q[i, j+1] + q[i, j-2]))

    for i in range(2, v_shape_0 - 2):
        for j in range(v_shape_1):
            F_y[i, j] = v[i, j] / 12 * (7 * (
                q[i, j] + q[i-1, j]) - (q[i+1, j] + q[i-2, j]))

    for i in range(q_shape_0):
        for j in range(q_shape_1):
            qout[i, j] = 0 # initialize empty qout
            if j >= 2 and j < q_shape_1 - 2:
                qout[i, j] = qout[i, j] - (F_x[i, j+1] - F_x[i, j]) / dx
            if i >= 2 and i < q_shape_0 - 2:
                qout[i, j] = qout[i, j] - (F_y[i+1, j] - F_y[i, j]) / dy

    cdef Py_ssize_t k
    for i in range(q_shape_0):
        for j in range(2):
            qout[i, j] = qout[i, j] - (
                (u_w[i, j] / dx) * (q[i, j+1] - q[i, j]) +
                (q[i, j] / dx) * (u[i, j+1] - u[i, j]))

            k = q_shape_1 - 2 + j  # like [-2:]
            qout[i, k] = qout[i, k] - (
                (u_e[i, j] / dx) * (q[i, k] - q[i, k-1]) +  # j in u_e is not out of place
                (q[i, k] / dx) * (u[i, k+1] - u[i, k]))


    cdef Py_ssize_t l
    for i in range(2):
        for j in range(q_shape_1):
            qout[i, j] = qout[i, j] - (
                (v_s[i, j] / dy) * (q[i+1, j] - q[i, j])
                + (q[i, j] / dy) * (v[i+1, j] - v[i, j]))

            l = q_shape_0 - 2 + i
            qout[l, j] = qout[l, j] - (
                (v_n[i, j] / dy) * (q[l, j] - q[l-1, j])
                + (q[l, j] / dy) * (v[l+1, j] - v[l, j]))


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.cdivision_warnings(True)
cpdef void cython_time_deriv_3(
    double [:, ::1] q, double [:, ::1] qint,
    double [:, ::1] u, double [:, ::1] v,
    double [:, ::1] qout, double [:, ::1] F_x, double [:, ::1] F_y,
    int dx, int dy, double dt,
    double [:, ::1] u_w, double [:, ::1] u_e,
    double [:, ::1] v_n, double [:, ::1] v_s) nogil:

    cdef Py_ssize_t q_shape_0 = q.shape[0]
    cdef Py_ssize_t q_shape_1 = q.shape[1]
    cdef Py_ssize_t i
    cdef Py_ssize_t j

    cython_space_deriv_4(q, u, v, qout, F_x, F_y, dx, dy, u_w, u_e, v_n, v_s)

    for i in range(q_shape_0):
        for j in range(q_shape_1):
            qint[i, j] = q[i, j] + dt / 3 * qout[i, j]
    cython_space_deriv_4(qint, u, v, qout, F_x, F_y, dx, dy, u_w, u_e, v_n, v_s)

    for i in range(q_shape_0):
        for j in range(q_shape_1):
            qint[i, j] = q[i, j] + dt / 2 * qout[i, j]
    cython_space_deriv_4(qint, u, v, qout, F_x, F_y, dx, dy, u_w, u_e, v_n, v_s)

    for i in range(q_shape_0):
        for j in range(q_shape_1):
            qout[i, j] = q[i, j] + dt * qout[i, j]
