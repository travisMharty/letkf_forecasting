import cv2
import numpy as np
from scipy import ndimage


def coarsen(array, coarseness):
    old_shape = np.array(array.shape, dtype=float)
    new_shape = coarseness * np.ceil(old_shape / coarseness).astype(int)

    row_add = int(new_shape[0] - old_shape[0])
    col_add = int(new_shape[1] - old_shape[1])
    padded = np.pad(array, ((0, row_add), (0, col_add)), mode='edge')
    temp = padded.reshape(new_shape[0] // coarseness, coarseness,
                          new_shape[1] // coarseness, coarseness)
    array = np.sum(temp, axis=(1, 3))/coarseness**2
    return array


def optical_flow(image0, image1, time0, time1, u, v):
    var_size = 7
    var_sig = 2
    var_thresh = 300
    sd_num = 2                  # for removing u_of & v_of
    coarseness = 4
    feature_params = dict(maxCorners=5000,
                          qualityLevel=0.0001,
                          minDistance=10,
                          blockSize=4)
    winSize = (50, 50)
    maxLevel = 5
    lk_params = dict(winSize=winSize,
                     maxLevel=maxLevel,
                     criteria=(cv2.TERM_CRITERIA_EPS |
                               cv2.TERM_CRITERIA_COUNT, 50, 0.03))

    image0 = coarsen(array=image0, coarseness=coarseness)
    image1 = coarsen(array=image1, coarseness=coarseness)

    c_min = min(image0.min(), image1.min())
    c_max = max(image0.max(), image1.max())
    image0 = (image0 - c_min)/(c_max - c_min)*255
    image1 = (image1 - c_min)/(c_max - c_min)*255
    image0 = image0.astype('uint8')
    image1 = image1.astype('uint8')

    u = coarsen(array=u, coarseness=coarseness)
    v = coarsen(array=v, coarseness=coarseness)
    U_median = np.median(u)
    V_median = np.median(v)

    x_step = (time1 - time0).seconds*U_median/(250*coarseness)
    y_step = (time1 - time0).seconds*V_median/(250*coarseness)
    x_step = int(np.round(x_step))
    y_step = int(np.round(y_step))

    p0 = cv2.goodFeaturesToTrack(
        image0,
        **feature_params)
    p0_resh = p0.reshape(p0.shape[0], p0.shape[2])
    means = ndimage.filters.uniform_filter(image0.astype('float'),
                                           (var_size, var_size))
    second_moments = ndimage.filters.uniform_filter(image0.astype('float')**2,
                                                    (var_size, var_size))
    variances = second_moments - means**2
    win_vars = ndimage.filters.gaussian_filter(variances, sigma=var_sig)
    win_vars = win_vars[
        (p0[:, :, 1].astype('int'), p0[:, :, 0].astype('int'))].ravel()

    p0 = p0[win_vars > var_thresh]
    p0_resh = p0.reshape(p0.shape[0], p0.shape[2])
    if p0_resh.size == 0:
        nothing = np.array([])
        return nothing, nothing, nothing

    p1_guess = p0 + np.array([x_step, y_step])[None, None, :]
    p1, status, err = cv2.calcOpticalFlowPyrLK(
        image0, image1, p0, p1_guess, **lk_params)

    status = status.ravel().astype('bool')
    p1 = p1[status, :, :]
    p0 = p0[status, :, :]

    # assumes clouds0 is square
    in_domain = np.logical_and(p1 > 0, p1 < image0.shape[0]).all(axis=-1)
    in_domain = in_domain.ravel()
    p1 = p1[in_domain, :, :]
    p0 = p0[in_domain, :, :]

    err = err.ravel()[status]
    p0_resh = p0.reshape(p0.shape[0], p0.shape[2])
    p1_resh = p1.reshape(p1.shape[0], p1.shape[2])

    time_step0 = (time1 - time0).seconds

    u_of = (p1_resh[:, 0] - p0_resh[:, 0])*(250*coarseness/(time_step0))
    v_of = (p1_resh[:, 1] - p0_resh[:, 1])*(250*coarseness/(time_step0))

    u_mu = u_of.mean()
    u_sd = np.sqrt(u_of.var())
    v_mu = v_of.mean()
    v_sd = np.sqrt(v_of.var())
    good_wind = ((u_of > u_mu - u_sd*sd_num) & (u_of < u_mu + u_sd*sd_num) &
                 (v_of > v_mu - v_sd*sd_num) & (v_of < v_mu + v_sd*sd_num))
    u_of = u_of[good_wind]
    v_of = v_of[good_wind]
    p1_good = p1_resh[good_wind]
    p1_good = np.round(p1_good)
    p1_good = p1_good.astype('int')
    return u_of, v_of, p1_good
