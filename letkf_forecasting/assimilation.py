import numpy as np
import scipy as sp
import letkf_forecasting.assimilation_accessories as assimilation_accessories
from scipy import interpolate


def optimal_interpolation(background, b_sig,
                          observations, o_sig,
                          P_field,
                          flat_locations, localization, sat_inflation,
                          spacial_localization=False, x=None, y=None):

    if spacial_localization:
        PHT = np.sqrt((x[:, None] - x[None, flat_locations])**2 +
                      (y[:, None] - y[None, flat_locations])**2)
        PHT = np.exp(-PHT/(localization))
        PHT = sat_inflation*PHT*b_sig**2

    if P_field is not None:
        PHT = np.abs(P_field[:, None] - P_field[None, flat_locations])
        PHT = np.exp(-PHT/(localization))
        PHT = sat_inflation*PHT*b_sig**2
    K = sp.linalg.solve(
        (PHT[flat_locations, :] + o_sig**2*np.eye(flat_locations.size)),
        PHT.T).T
    to_return = background + K.dot(observations - background[flat_locations])

    return to_return


def reduced_cov(ensemble, locations):
    this_ens = ensemble.copy()
    Ne = this_ens.shape[1]
    mu = this_ens.mean(axis=1)
    this_ens -= mu[:, None]
    cov = this_ens.dot(this_ens[locations].T)
    cov = cov/(Ne - 1)
    return cov


def reduced_enkf(ensemble,
                 observations, R_sig,
                 flat_locations, inflation,
                 localization=0, x=None, y=None):
    ens_num = ensemble.shape[1]
    obs_size = observations.size
    PHT = reduced_cov(ensemble, flat_locations)
    if localization > 1e-16:
        rhoHT = ((x[:, None] - x[None, flat_locations])**2 +
                 (y[:, None] - y[None, flat_locations])**2)
        rhoHT = np.exp(-rhoHT/(2*localization**2))
        PHT = PHT*rhoHT
    K = sp.linalg.solve(
        (PHT[flat_locations, :] + R_sig**2*np.eye(flat_locations.size)),
        PHT.T).T
    rand_obs = np.random.normal(
        loc=0.0, scale=R_sig, size=ens_num*obs_size)
    rand_obs = rand_obs.reshape(obs_size, ens_num)
    rand_obs += observations[:, None]
    analysis = ensemble + K.dot(rand_obs - ensemble[flat_locations])
    return analysis


def assimilate(ensemble, observations, flat_sensor_indices, R_inverse,
               inflation, domain_shape=False,
               localization_length=False, assimilation_positions=False,
               assimilation_positions_2d=False,
               full_positions_2d=False):
    """
    *** NEED TO REWRITE
    Assimilates observations into ensemble using the LETKF.

    Parameters
    ----------
    ensemble : array
         The ensemble of size kxn where k is the number of ensemble members
         and n is the state vector size.
    observations : array
         An observation vector of length m.
    H : array
         Forward observation matrix of size mxn. **may need changing**
    R_inverse : array
         Inverse of observation error matrix. **will need changing**
    inflation : float
         Inflation parameter.
    localization_length : float
         Localization distance in each direction so that assimilation will take
         on (2*localization + 1)**2 elements. If equal to False then no
         localization will take place.
    assimilation_positions : array
         Row and column index of state domain over which assimilation will
         take place. First column contains row positions, second column
         contains column positions, total number of rows is number of
         assimilations. If False the assimilation will take place over
         full_positions. If localization_length is False then this variable
         will not be used.
    full_positions : array
         Array similar to assimilation_positions including the positions of
         all elements of the state.

    Return
    ------
    ensemble : array
         Analysis ensemble of the same size as input ensemble
    """
    # Change to allow for R to not be pre-inverted?
    if localization_length is False:

        # LETKF without localization
        Y_b = ensemble[flat_sensor_indices, :]
        y_b_bar = Y_b.mean(axis=1)
        Y_b -= y_b_bar[:, None]
        x_bar = ensemble.mean(axis=1)
        ensemble -= x_bar[:, None]
        ens_size = ensemble.shape[1]
        # C = (Y_b.T).dot(R_inverse)
        C = Y_b.T*R_inverse
        eig_value, eig_vector = np.linalg.eigh(
            (ens_size-1)*np.eye(ens_size)/inflation + C.dot(Y_b))
        P_tilde = eig_vector.copy()
        W_a = eig_vector.copy()
        for i, num in enumerate(eig_value):
            P_tilde[:, i] *= 1/num
            W_a[:, i] *= 1/np.sqrt(num)
        P_tilde = P_tilde.dot(eig_vector.T)
        W_a = W_a.dot(eig_vector.T)*(np.sqrt(ens_size - 1))
        w_a_bar = P_tilde.dot(C.dot(observations - y_b_bar))
        W_a += w_a_bar[:, None]
        ensemble = x_bar[:, None] + ensemble.dot(W_a)
        return ensemble

    else:
        # LETKF with localization assumes H is I
        # NEED: to include wind in ensemble will require reworking due to
        # new H and different localization.
        # NEED: Change to include some form of H for paralax correction??
        # Maybe: ^ not if paralax is only corrected when moving ...
        # to ground sensors.
        # SHOULD: Will currently write as though R_inverse is a scalar.
        # May need to change at some point but will likely need to do
        # something clever since R_inverse.size is 400 billion
        # best option: form R_inverse inside of localization routine
        # good option: assimilate sat images at low resolution ...
        # (probabily should do this either way)
        x_bar = ensemble.mean(axis=1)  # Need to bring this back
        ensemble -= x_bar[:, None]
        ens_size = ensemble.shape[1]
        kal_count = 0
        W_interp = np.zeros([assimilation_positions.size, ens_size**2])
        for interp_position in assimilation_positions:
            local_positions = assimilation_accessories.nearest_positions(
                interp_position, domain_shape,
                localization_length)
            local_ensemble = ensemble[local_positions]
            local_x_bar = x_bar[local_positions]
            local_obs = observations[local_positions]  # assume H is I
            # assume R_inverse is diag*const
            C = (local_ensemble.T)*R_inverse

            # This should be better, but I can't get it to work
            eig_value, eig_vector = np.linalg.eigh(
                (ens_size-1)*np.eye(ens_size)/inflation +
                C.dot(local_ensemble))
            P_tilde = eig_vector.copy()
            W_a = eig_vector.copy()
            for i, num in enumerate(eig_value):
                P_tilde[:, i] *= 1/num
                W_a[:, i] *= 1/np.sqrt(num)
            P_tilde = P_tilde.dot(eig_vector.T)
            W_a = W_a.dot(eig_vector.T)*np.sqrt(ens_size - 1)

            # P_tilde = np.linalg.inv(
            #     (ens_size - 1)*np.eye(ens_size)/inflation +
            #     C.dot(local_ensemble))
            # W_a = np.real(sp.linalg.sqrtm((ens_size - 1)*P_tilde))
            w_a_bar = P_tilde.dot(C.dot(local_obs - local_x_bar))
            W_a += w_a_bar[:, None]
            W_interp[kal_count] = np.ravel(W_a)  # separate w_bar??
            kal_count += 1
        if assimilation_positions_2d.size != full_positions_2d.size:
            W_fun = interpolate.LinearNDInterpolator(assimilation_positions_2d,
                                                     W_interp)
            W_fine_mesh = W_fun(full_positions_2d)
            W_fine_mesh = W_fine_mesh.reshape(domain_shape[0]*domain_shape[1],
                                              ens_size, ens_size)
        else:
            W_fine_mesh = W_interp.reshape(domain_shape[0]*domain_shape[1],
                                           ens_size, ens_size)
        ensemble = x_bar[:, None] + np.einsum(
            'ij, ijk->ik', ensemble, W_fine_mesh)

        return ensemble


def assimilate_full_wind(ensemble, observations, flat_sensor_indices,
                         R_inverse, R_inverse_wind, inflation, wind_inflation,
                         domain_shape=False, U_shape=False, V_shape=False,
                         localization_length=False,
                         localization_length_wind=False,
                         assimilation_positions=False,
                         assimilation_positions_2d=False,
                         full_positions_2d=False):
    # seperate out localization in irradiance and wind
    """
    *** NEED TO REWRITE Documentation
    Assimilates observations into ensemble using the LETKF.

    Parameters
    ----------
    ensemble : array
         The ensemble of size kxn where k is the number of ensemble members
         and n is the state vector size.
    observations : array
         An observation vector of length m.
    H : array
         Forward observation matrix of size mxn. **may need changing**
    R_inverse : array
         Inverse of observation error matrix. **will need changing**
    inflation : float
         Inflation parameter.
    localization_length : float
         Localization distance in each direction so that assimilation will take
         on (2*localization + 1)**2 elements. If equal to False then no
         localization will take place.
    assimilation_positions : array
         Row and column index of state domain over which assimilation will
         take place. First column contains row positions, second column
         contains column positions, total number of rows is number of
         assimilations. If False the assimilation will take place over
         full_positions. If localization_length is False then this variable
         will not be used.
    full_positions : array
         Array similar to assimilation_positions including the positions of
         all elements of the state.

    Return
    ------
    ensemble : array
         Analysis ensemble of the same size as input ensemble
    """

    # LETKF with localization assumes H is I
    # NEED: to include wind in ensemble will require reworking due to
    # new H and different localization.
    # NEED: Change to include some form of H for paralax correction??
    # Maybe: ^ not if paralax is only corrected when moving to ground sensors.
    # SHOULD: Will currently write as though R_inverse is a scalar.
    # May need to change at some point but will likely need to do
    # something clever since R_inverse.size is 400 billion
    # best option: form R_inverse inside of localization routine
    # good option: assimilate sat images at low resolution (probabily should...
    # do this either way)

    U_size = U_shape[0]*U_shape[1]
    V_size = V_shape[0]*V_shape[1]
    ensemble_U = ensemble[:U_size]
    x_bar_U = ensemble_U.mean(axis=1)
    ensemble_U -= x_bar_U[:, None]
    ensemble_V = ensemble[U_size:V_size + U_size]
    x_bar_V = ensemble_V.mean(axis=1)
    ensemble_V -= x_bar_V[:, None]
    ensemble_csi = ensemble[U_size + V_size:]
    x_bar_csi = ensemble_csi.mean(axis=1)
    ensemble_csi -= x_bar_csi[:, None]
    ens_size = ensemble.shape[1]
    kal_count = 0
    W_interp = np.zeros([assimilation_positions.size, ens_size**2])
    W_interp_wind = W_interp.copy()*np.nan
    # bad_count = 0
    for interp_position in assimilation_positions:
        # for the irradiance portion of the ensemble
        local_positions = assimilation_accessories.nearest_positions(
            interp_position, domain_shape,
            localization_length)
        local_ensemble = ensemble_csi[local_positions]
        local_x_bar = x_bar_csi[local_positions]
        local_obs = observations[local_positions]  # assume H is I
        C = (local_ensemble.T)*R_inverse  # assume R_inverse is diag+const

        eig_value, eig_vector = np.linalg.eigh(
            (ens_size-1)*np.eye(ens_size)/inflation + C.dot(local_ensemble))
        P_tilde = eig_vector.copy()
        W_a = eig_vector.copy()
        for i, num in enumerate(eig_value):
            P_tilde[:, i] *= 1/num
            W_a[:, i] *= 1/np.sqrt(num)
        P_tilde = P_tilde.dot(eig_vector.T)
        W_a = W_a.dot(eig_vector.T)*np.sqrt(ens_size - 1)
        w_a_bar = P_tilde.dot(C.dot(local_obs - local_x_bar))
        W_a += w_a_bar[:, None]
        W_interp[kal_count] = np.ravel(W_a)  # separate w_bar??

        # should eventually change to assimilate on coarser wind grid
        local_positions = assimilation_accessories.nearest_positions(
            interp_position, domain_shape,
            localization_length_wind)
        local_ensemble = ensemble_csi[local_positions]
        local_x_bar = x_bar_csi[local_positions]
        local_obs = observations[local_positions]  # assume H is I
        C = (local_ensemble.T)*R_inverse_wind  # assume R_inverse is diag+const
        eig_value, eig_vector = np.linalg.eigh(
            (ens_size-1)*np.eye(ens_size)/wind_inflation +
            C.dot(local_ensemble))
        P_tilde = eig_vector.copy()
        W_a = eig_vector.copy()
        for i, num in enumerate(eig_value):
            P_tilde[:, i] *= 1/num
            W_a[:, i] *= 1/np.sqrt(num)
        P_tilde = P_tilde.dot(eig_vector.T)
        W_a = W_a.dot(eig_vector.T)*np.sqrt(ens_size - 1)
        w_a_bar = P_tilde.dot(C.dot(local_obs - local_x_bar))
        W_a += w_a_bar[:, None]
        W_interp_wind[kal_count] = np.ravel(W_a)  # separate w_bar??
        kal_count += 1
    if assimilation_positions_2d.size != full_positions_2d.size:
        W_fun = interpolate.LinearNDInterpolator(assimilation_positions_2d,
                                                 W_interp)
        W_interp = W_fun(full_positions_2d)

        W_fun = interpolate.LinearNDInterpolator(assimilation_positions_2d,
                                                 W_interp_wind)
        W_interp_wind = W_fun(full_positions_2d)

    W_fine_mesh = W_interp.reshape(domain_shape[0]*domain_shape[1],
                                   ens_size, ens_size)
    # change this to its own variable
    W_interp = W_interp_wind.reshape(domain_shape[0], domain_shape[1],
                                     ens_size, ens_size)
    ensemble_csi = x_bar_csi[:, None] + np.einsum(
        'ij, ijk->ik', ensemble_csi, W_fine_mesh)
    W_fine_mesh = (np.pad(W_interp, [(0, 0), (0, 1), (0, 0), (0, 0)],
                          mode='edge')).reshape(U_size, ens_size, ens_size)
    ensemble_U = x_bar_U[:, None] + np.einsum(
        'ij, ijk->ik', ensemble_U, W_fine_mesh)
    W_fine_mesh = (np.pad(W_interp, [(0, 1), (0, 0), (0, 0), (0, 0)],
                          mode='edge')).reshape(V_size, ens_size, ens_size)
    ensemble_V = x_bar_V[:, None] + np.einsum(
        'ij, ijk->ik', ensemble_V, W_fine_mesh)
    ensemble = np.concatenate([ensemble_U, ensemble_V, ensemble_csi], axis=0)

    return ensemble


def assimilate_sat_to_wind(ensemble, observations,
                           R_inverse_wind, wind_inflation,
                           domain_shape=False, U_shape=False, V_shape=False,
                           localization_length_wind=False,
                           assimilation_positions=False,
                           assimilation_positions_2d=False,
                           full_positions_2d=False):
    # separate out localization in irradiance and wind
    """
    *** NEED TO REWRITE Documentation
    Assimilates observations into ensemble using the LETKF.

    Parameters
    ----------
    ensemble : array
         The ensemble of size kxn where k is the number of ensemble members
         and n is the state vector size.
    observations : array
         An observation vector of length m.
    H : array
         Forward observation matrix of size mxn. **may need changing**
    R_inverse : array
         Inverse of observation error matrix. **will need changing**
    inflation : float
         Inflation parameter.
    localization_length : float
         Localization distance in each direction so that assimilation will take
         on (2*localization + 1)**2 elements. If equal to False then no
         localization will take place.
    assimilation_positions : array
         Row and column index of state domain over which assimilation will
         take place. First column contains row positions, second column
         contains column positions, total number of rows is number of
         assimilations. If False the assimilation will take place over
         full_positions. If localization_length is False then this variable
         will not be used.
    full_positions : array
         Array similar to assimilation_positions including the positions of
         all elements of the state.

    Return
    ------
    ensemble : array
         Analysis ensemble of the same size as input ensemble
    """

    # LETKF with localization assumes H is I
    # NEED: to include wind in ensemble will require reworking due to
    # new H and different localization.
    # NEED: Change to include some form of H for paralax correction??
    # Maybe: ^ not if paralax is only corrected when moving to ground sensors.
    # SHOULD: Will currently write as though R_inverse is a scalar.
    # May need to change at some point but will likely need to do
    # something clever since R_inverse.size is 400 billion
    # best option: form R_inverse inside of localization routine
    # good option: assimilate sat images at low resolution ...
    # (probabily should do this either way)

    U_size = U_shape[0]*U_shape[1]
    V_size = V_shape[0]*V_shape[1]
    ensemble_U = ensemble[:U_size]
    x_bar_U = ensemble_U.mean(axis=1)
    ensemble_U -= x_bar_U[:, None]
    ensemble_V = ensemble[U_size:V_size + U_size]
    x_bar_V = ensemble_V.mean(axis=1)
    ensemble_V -= x_bar_V[:, None]
    ensemble_csi = ensemble[U_size + V_size:]
    x_bar_csi = ensemble_csi.mean(axis=1)
    ensemble_csi -= x_bar_csi[:, None]
    ens_size = ensemble.shape[1]
    kal_count = 0
    W_interp = np.zeros([assimilation_positions.size, ens_size**2])
    W_interp_wind = W_interp.copy()*np.nan
    # bad_count = 0
    for interp_position in assimilation_positions:
        # # for the irradiance portion of the ensemble
        # local_positions = assimilation_accessories.nearest_positions(
        #     interp_position, domain_shape,
        #     localization_length)
        # local_ensemble = ensemble_csi[local_positions]
        # local_x_bar = x_bar_csi[local_positions]
        # local_obs = observations[local_positions] # assume H is I
        # C = (local_ensemble.T)*R_inverse  # assume R_inverse is diag+const

        # eig_value, eig_vector = np.linalg.eigh(
        #     (ens_size-1)*np.eye(ens_size)/inflation + C.dot(local_ensemble))
        # P_tilde = eig_vector.copy()
        # W_a = eig_vector.copy()
        # for i, num in enumerate(eig_value):
        #     P_tilde[:, i] *= 1/num
        #     W_a[:, i] *= 1/np.sqrt(num)
        # P_tilde = P_tilde.dot(eig_vector.T)
        # W_a = W_a.dot(eig_vector.T)*np.sqrt(ens_size - 1)
        # w_a_bar = P_tilde.dot(C.dot(local_obs - local_x_bar))
        # W_a += w_a_bar[:, None]
        # W_interp[kal_count] = np.ravel(W_a) # separate w_bar??

        # should eventually change to assimilate on coarser wind grid
        local_positions = assimilation_accessories.nearest_positions(
            interp_position, domain_shape,
            localization_length_wind)
        local_ensemble = ensemble_csi[local_positions]
        local_x_bar = x_bar_csi[local_positions]
        local_obs = observations[local_positions]  # assume H is I
        C = (local_ensemble.T)*R_inverse_wind  # assume R_inverse is diag+const
        eig_value, eig_vector = np.linalg.eigh(
            (ens_size-1)*np.eye(ens_size)/wind_inflation +
            C.dot(local_ensemble))
        P_tilde = eig_vector.copy()
        W_a = eig_vector.copy()
        for i, num in enumerate(eig_value):
            P_tilde[:, i] *= 1/num
            W_a[:, i] *= 1/np.sqrt(num)
        P_tilde = P_tilde.dot(eig_vector.T)
        W_a = W_a.dot(eig_vector.T)*np.sqrt(ens_size - 1)
        w_a_bar = P_tilde.dot(C.dot(local_obs - local_x_bar))
        W_a += w_a_bar[:, None]
        W_interp_wind[kal_count] = np.ravel(W_a)  # separate w_bar??
        kal_count += 1
    if assimilation_positions_2d.size != full_positions_2d.size:
        W_fun = interpolate.LinearNDInterpolator(assimilation_positions_2d,
                                                 W_interp_wind)
        W_interp_wind = W_fun(full_positions_2d)
    # change this to its own variable
    W_interp = W_interp_wind.reshape(domain_shape[0], domain_shape[1],
                                     ens_size, ens_size)
    W_fine_mesh = (np.pad(W_interp, [(0, 0), (0, 1), (0, 0), (0, 0)],
                          mode='edge')).reshape(U_size, ens_size, ens_size)
    ensemble_U = x_bar_U[:, None] + np.einsum(
        'ij, ijk->ik', ensemble_U, W_fine_mesh)
    W_fine_mesh = (np.pad(W_interp, [(0, 1), (0, 0), (0, 0), (0, 0)],
                          mode='edge')).reshape(V_size, ens_size, ens_size)
    ensemble_V = x_bar_V[:, None] + np.einsum(
        'ij, ijk->ik', ensemble_V, W_fine_mesh)
    # leave csi unchanged
    ensemble_csi += x_bar_csi[:, None]

    ensemble = np.concatenate([ensemble_U, ensemble_V, ensemble_csi], axis=0)

    return ensemble


def assimilate_wrf(ensemble, observations,
                   R_inverse, wind_inflation,
                   wind_shape,
                   localization_length_wind=False,
                   assimilation_positions=False,
                   assimilation_positions_2d=False,
                   full_positions_2d=False):
    # Currently doing U and V separately
    """
    *** NEED TO REWRITE Documentation
    Assimilates observations into ensemble using the LETKF.

    Parameters
    ----------
    ensemble : array
         The ensemble of size kxn where k is the number of ensemble members
         and n is the state vector size.
    observations : array
         An observation vector of length m.
    H : array
         Forward observation matrix of size mxn. **may need changing**
    R_inverse : array
         Inverse of observation error matrix. **will need changing**
    inflation : float
         Inflation parameter.
    localization_length : float
         Localization distance in each direction so that assimilation will take
         on (2*localization + 1)**2 elements. If equal to False then no
         localization will take place.
    assimilation_positions : array
         Row and column index of state domain over which assimilation will
         take place. First column contains row positions, second column
         contains column positions, total number of rows is number of
         assimilations. If False the assimilation will take place over
         full_positions. If localization_length is False then this variable
         will not be used.
    full_positions : array
         Array similar to assimilation_positions including the positions of
         all elements of the state.

    Return
    ------
    ensemble : array
         Analysis ensemble of the same size as input ensemble
    """

    # LETKF with localization assumes H is I
    # NEED: to include wind in ensemble will require reworking due to
    # new H and different localization.
    # NEED: Change to include some form of H for paralax correction??
    # Maybe: ^ not if paralax is only corrected when moving to ground sensors.
    # SHOULD: Will currently write as though R_inverse is a scalar.
    # May need to change at some point but will likely need to do
    # something clever since R_inverse.size is 400 billion
    # best option: form R_inverse inside of localization routine
    # good option: assimilate sat images at low resolution...
    # (probabily should do this either way)

    x_bar = ensemble.mean(axis=1)
    ensemble -= x_bar[:, None]
    ens_size = ensemble.shape[1]
    kal_count = 0
    W_interp = np.zeros([assimilation_positions.size, ens_size**2])
    for interp_position in assimilation_positions:
        # for the irradiance portion of the ensemble
        local_positions = assimilation_accessories.nearest_positions(
            interp_position, wind_shape,
            localization_length_wind)
        local_ensemble = ensemble[local_positions]
        local_x_bar = x_bar[local_positions]
        local_obs = observations[local_positions]  # assume H is I
        C = (local_ensemble.T)*R_inverse  # assume R_inverse is diag+const
        eig_value, eig_vector = np.linalg.eigh(
            (ens_size-1)*np.eye(ens_size)/wind_inflation +
            C.dot(local_ensemble))
        P_tilde = eig_vector.copy()
        W_a = eig_vector.copy()
        for i, num in enumerate(eig_value):
            P_tilde[:, i] *= 1/num
            W_a[:, i] *= 1/np.sqrt(num)
        P_tilde = P_tilde.dot(eig_vector.T)
        W_a = W_a.dot(eig_vector.T)*np.sqrt(ens_size - 1)
        w_a_bar = P_tilde.dot(C.dot(local_obs - local_x_bar))
        W_a += w_a_bar[:, None]
        W_interp[kal_count] = np.ravel(W_a)  # separate w_bar??
        kal_count += 1
    if assimilation_positions_2d.size != full_positions_2d.size:
        W_fun = sp.interpolate.LinearNDInterpolator(assimilation_positions_2d,
                                                    W_interp)
        W_interp = W_fun(full_positions_2d)

    W_fine_mesh = W_interp.reshape(wind_shape[0]*wind_shape[1],
                                   ens_size, ens_size)
    ensemble = x_bar[:, None] + np.einsum(
        'ij, ijk->ik', ensemble, W_fine_mesh)

    return ensemble


def assimilate_wind(ensemble, observations, flat_sensor_indices, R_inverse,
                    inflation, wind_size):
    """
    *** NEED TO REWRITE
    Assimilates observations into ensemble using the LETKF.

    Parameters
    ----------
    ensemble : array
         The ensemble of size kxn where k is the number of ensemble members
         and n is the state vector size.
    observations : array
         An observation vector of length m
    R_inverse : array
         Inverse of observation error matrix. **will need changing**
    inflation : float
         Inflation parameter

    Return
    ------
    winds
    """

    # LETKF without localization
    Y_b = ensemble[wind_size:].copy()
    y_b_bar = Y_b.mean(axis=1)
    Y_b -= y_b_bar[:, None]
    x_bar = ensemble.mean(axis=1)  # Need to bring this back
    ensemble -= x_bar[:, None]
    ens_size = ensemble.shape[1]
    C = Y_b.T*R_inverse
    eig_value, eig_vector = np.linalg.eigh(
        (ens_size-1)*np.eye(ens_size)/inflation + C.dot(Y_b))
    P_tilde = eig_vector.copy()
    W_a = eig_vector.copy()
    for i, num in enumerate(eig_value):
        P_tilde[:, i] *= 1/num
        W_a[:, i] *= 1/np.sqrt(num)
    P_tilde = P_tilde.dot(eig_vector.T)
    W_a = W_a.dot(eig_vector.T)*(np.sqrt(ens_size - 1))
    w_a_bar = P_tilde.dot(C.dot(observations - y_b_bar))
    W_a += w_a_bar[:, None]
    ensemble = x_bar[:, None] + ensemble.dot(W_a)
    return ensemble[:wind_size]
