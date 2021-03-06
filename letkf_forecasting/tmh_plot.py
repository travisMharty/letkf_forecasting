import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import letkf_forecasting.analyse_results as ar


def of_vectors_plot(clouds0_8b, clouds1_8b, p0_good, p1_good, u, v,
                    time0, time1, adjust):
    adjust = 1.1
    nrows, ncols = 1, 2
    dy, dx = clouds0_8b.shape
    figsize = plt.figaspect(float(dy * nrows) / float(adjust * dx * ncols))
    fig, ax = plt.subplots(nrows, ncols, sharey=True, sharex=True,
                           figsize=figsize, dpi=300)
    im = ax[0].pcolormesh(clouds0_8b,
                          cmap='Blues')
    ax[0].scatter(p0_good[:, 0], p0_good[:, 1],
                  c='r', alpha=.5)
    qu = ax[0].quiver(p0_good[:, 0], p0_good[:, 1], u, v)
    ax[0].set_aspect('equal', 'datalim')
    ax[0].set_title(time0)
    ax[0].set_xlabel('Distance (km)')
    ax[0].set_ylabel('Distance (km)')

    im = ax[1].pcolormesh(clouds1_8b,
                          cmap='Blues')
    ax[1].scatter(p1_good[:, 0], p1_good[:, 1],
                  c='r', alpha=.5)
    ax[1].set_aspect('equal', 'datalim')
    ax[1].set_title(time1)
    ax[1].set_xlabel('Distance (km)')
    ax[0].set(ylim=[0, dy])
    ax[0].set(xlim=[0, dx])
    ax[1].set(ylim=[0, dy])
    ax[1].set(xlim=[0, dx])
    return fig, ax


def wrf_opt_flow_plot(ensemble, U_shape, V_shape,
                      U_of, V_of, of_coord,
                      adjust, cmap='bwr'):
    fraction = 0.10
    pad = 0.02
    nrows, ncols = 1, 2
    dy, dx = U_shape
    U_size = U_shape[0]*U_shape[1]
    wind_size = U_size + V_shape[0]*V_shape[1]
    U = ensemble[:U_size].mean(axis=1).reshape(U_shape)
    V = ensemble[U_size:wind_size].mean(axis=1).reshape(V_shape)
    figsize = plt.figaspect(float(dy * nrows) / float(adjust * dx * ncols))
    fig, ax = plt.subplots(nrows, ncols, sharey=True, sharex=True,
                           figsize=figsize, dpi=300)
    vmax = np.max([np.abs(U).max(), np.abs(V).max(),
                   np.abs(U_of).max(), np.abs(V_of).max()])
    nc = 15
    bounds = np.linspace(-vmax, vmax, nc)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    im = ax[0].pcolormesh(U,
                          cmap=cmap,
                          norm=norm)
    ax[0].scatter(of_coord[:, 0], of_coord[:, 1], c=U_of,
                  cmap=cmap,
                  norm=norm)
    ax[0].set_aspect('equal', 'datalim')
    ax[0].set_title('U')
    ax[0].set_xlabel('Position (km/4)')
    ax[0].set_ylabel('Position (km/4)')

    im = ax[1].pcolormesh(V,
                          cmap=cmap,
                          norm=norm)
    ax[1].scatter(of_coord[:, 0], of_coord[:, 1], c=V_of,
                  cmap=cmap,
                  norm=norm)
    ax[1].set_aspect('equal', 'datalim')
    ax[1].set_title('V')
    ax[1].set_xlabel('Position (km/4)')
    cb = plt.colorbar(im, ax=ax.tolist(), pad=pad,
                      fraction=fraction, label='Wind Speed (m/s)')
    cb.ax.set_ylabel('Wind Speed (m/s)')
    ax[0].set(xlim=[0, dx])
    ax[0].set(ylim=[0, dy])
    ax[1].set(xlim=[0, dx])
    ax[1].set(ylim=[0, dy])

    return fig, ax


def opt_flow_sd_plot(ensemble, U_shape, V_shape,
                     U_of, V_of, of_coord,
                     adjust, cmap='Greys'):
    fraction = 0.10
    pad = 0.02
    nrows, ncols = 1, 2
    dy, dx = U_shape
    U_size = U_shape[0]*U_shape[1]
    wind_size = U_size + V_shape[0]*V_shape[1]
    U = np.sqrt(ensemble[:U_size].var(axis=1).reshape(U_shape))
    V = np.sqrt(ensemble[U_size:wind_size].var(axis=1).reshape(V_shape))
    figsize = plt.figaspect(float(dy * nrows) / float(adjust * dx * ncols))
    fig, ax = plt.subplots(nrows, ncols, sharey=True, sharex=True,
                           figsize=figsize, dpi=300)
    vmax = np.max([np.abs(U).max(), np.abs(V).max()])
    vmin = 0
    nc = 15
    bounds = np.linspace(vmin, vmax, nc)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    im = ax[0].pcolormesh(U,
                          cmap=cmap,
                          norm=norm)
    ax[0].scatter(of_coord[:, 0], of_coord[:, 1],
                  cmap=cmap,
                  norm=norm)
    ax[0].set_aspect('equal', 'datalim')
    ax[0].set_title('U_sd')
    ax[0].set_xlabel('Position (km/4)')
    ax[0].set_ylabel('Position (km/4)')

    im = ax[1].pcolormesh(V,
                          cmap=cmap,
                          norm=norm)
    ax[1].scatter(of_coord[:, 0], of_coord[:, 1],
                  cmap=cmap,
                  norm=norm)
    ax[1].set_aspect('equal', 'datalim')
    ax[1].set_title('V_var')
    ax[1].set_xlabel('Position (km/4)')
    cb = plt.colorbar(im, ax=ax.tolist(), pad=pad,
                      fraction=fraction, label='Wind Speed (m/s)')
    cb.ax.set_ylabel('Wind Speed (m/s)')
    ax[0].set(xlim=[0, dx])
    ax[0].set(ylim=[0, dy])
    ax[1].set(xlim=[0, dx])
    ax[1].set(ylim=[0, dy])

    return fig, ax


def var_plot(ensemble, domain_shape,
             adjust, cmap='Greys', vmin=0, vmax=None):
    var = ensemble.var(axis=1)
    if vmax is None:
        vmax = var.max()
    nc = 11
    bounds = np.linspace(vmin, vmax, nc)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    fraction = 0.10
    pad = 0.02
    dy, dx = domain_shape
    figsize = plt.figaspect(float(dy) /
                            (adjust*(1 + fraction + pad)*float(dx)))
    fig, ax = plt.subplots(sharey=True, sharex=True,
                           figsize=figsize, dpi=150)
    im = ax.pcolormesh(var.reshape(domain_shape),
                       vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)
    plt.colorbar(im)
    ax.set(xlim=[0, domain_shape[1]], ylim=[0, domain_shape[0]])
    ax.set(aspect='equal', adjustable='box-forced')
    return fig, ax


def ensemble_stamps_cdf(
        others, other_titles, ensemble, nrows, ncols, domain_shape, adjust,
        cmap='Blues', vmin=0, vmax=1, nc=11, cbar_label=None, dpi=150):
    if vmin is None or vmax is None:
        vmax = ensemble.max()
        vmin = -vmax
    bounds = np.linspace(vmin, vmax, nc)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    fraction = 0.10
    pad = 0.02
    dy, dx = domain_shape
    figsize = plt.figaspect(float(dy * nrows) /
                            (adjust*(1 + fraction + pad)*float(dx * ncols)))
    fig, ax = plt.subplots(nrows, ncols,
                           sharey=True, sharex=True,
                           figsize=figsize, dpi=dpi)
    for j in range(len(other_titles)):
        if others[j].shape == domain_shape:
            this_other = others[j]
        else:
            this_other = others[j].reshape(domain_shape)
        im = ax[0, j].pcolormesh(
            this_other,
            cmap=cmap,
            norm=norm)
        ax[0, j].set_title(other_titles[j])
        ax[0, j].axis('off')
    try:
        j
    except Exception:
        j=-1
    im = ax[0, j + 1].pcolormesh(
        ar.return_ens_mean(ensemble),
        cmap=cmap,
        norm=norm)
    ax[0, j + 1].set_title('Mean')
    ax[0, j + 1].axis('off')

    ens_count = 0
    for jj in range(j + 2, ncols):
        ax[0, jj].pcolormesh(
            ensemble.isel(ensemble_number=ens_count).values,
            cmap=cmap,
            norm=norm)
        ax[0, jj].axis('off')
        ens_count += 1
    for i in range(1, nrows):
        for j in range(ncols):
            ax[i, j].pcolormesh(
                ensemble.isel(ensemble_number=ens_count).values,
                cmap=cmap,
                norm=norm)
            ax[i, j].axis('off')
            ens_count += 1

    for i in range(nrows):
        for j in range(ncols):
            ax[i, j].set(xlim=[0, domain_shape[1]], ylim=[0, domain_shape[0]])
            ax[i, j].set(aspect='equal', adjustable='box-forced')

    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.colorbar(im, ax=ax.ravel().tolist(), pad=pad, fraction=fraction,
                 label=cbar_label)
    return fig, ax


def ensemble_stamps(
        others, other_titles, ensemble, nrows, ncols, domain_shape, adjust,
        cmap='Blues', vmin=0, vmax=1, cbar_label=None):
    if vmin is None or vmax is None:
        vmax = np.max(np.abs(ensemble))
        vmin = -vmax
    nc = 11
    bounds = np.linspace(vmin, vmax, nc)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    fraction = 0.10
    pad = 0.02
    dy, dx = domain_shape
    figsize = plt.figaspect(float(dy * nrows) /
                            (adjust*(1 + fraction + pad)*float(dx * ncols)))
    fig, ax = plt.subplots(nrows, ncols,
                           sharey=True, sharex=True,
                           figsize=figsize, dpi=150)
    for j in range(len(other_titles)):
        if others[j].shape == domain_shape:
            this_other = others[j]
        else:
            this_other = others[j].reshape(domain_shape)
        im = ax[0, j].pcolormesh(
            this_other,
            cmap=cmap,
            norm=norm)
        ax[0, j].set_title(other_titles[j])
        ax[0, j].axis('off')
    try:
        j
    except Exception:
        j = -1
    im = ax[0, j + 1].pcolormesh(
        ensemble.mean(axis=1).reshape(domain_shape),
        cmap=cmap,
        norm=norm)
    ax[0, j + 1].axis('off')

    ens_count = 0
    for jj in range(j + 2, ncols):
        ax[0, jj].pcolormesh(
            ensemble[:, ens_count].reshape(domain_shape),
            cmap=cmap,
            norm=norm)
        ax[0, jj].axis('off')
        ens_count += 1
    for i in range(1, nrows):
        for j in range(ncols):
            ax[i, j].pcolormesh(
                ensemble[:, ens_count].reshape(domain_shape),
                cmap=cmap,
                norm=norm)
            ax[i, j].axis('off')
            ens_count += 1

    for i in range(nrows):
        for j in range(ncols):
            ax[i, j].set(xlim=[0, domain_shape[1]], ylim=[0, domain_shape[0]])
            ax[i, j].set(aspect='equal', adjustable='box-forced')

    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.colorbar(im, ax=ax.ravel().tolist(), pad=pad, fraction=fraction,
                 label=cbar_label)
    return fig, ax


def subplots(data, x, y, subplot_titles, axes_labels, sup_title=None,
             cb_label=None, cmap='Blues', vmax=None, adjust=1, nc=11):
    """
    data and subplot_titles are lists of the same length
    """
    fraction = 0.10
    pad = .02
    if vmax is None:
        vmin = 0
        vmax = 1
    else:
        vmin = -vmax
    bounds = np.linspace(vmin, vmax, nc)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    suptitle_x = .5*(1 - fraction)
    nrows, ncols = 1, len(data)
    dy, dx = data[0].shape
    if cb_label is None:
        figsize = plt.figaspect(float(dy * nrows) / float(adjust*dx * ncols))
    else:
        figsize = plt.figaspect(float(dy * nrows) /
                                ((1 + fraction + pad)*float(adjust*dx * ncols)))
    fig, ax = plt.subplots(nrows, ncols, sharey=True, sharex=True,
                           figsize=figsize, dpi=300)
    if sup_title is not None:
        plt.suptitle(sup_title, x=suptitle_x)
    for j in range(ncols):
        im = ax[j].pcolormesh(
                x, y,
                data[j],
                cmap=cmap, norm=norm)
        ax[j].set_title(subplot_titles[j])
    if cb_label is not None:
        cb = plt.colorbar(im, ax=ax.ravel().tolist(),
                          pad=pad, fraction=fraction)
        cb.set_label(cb_label)
    ax[0].set_ylabel(axes_labels[1])
    for j in range(ncols):
        ax[j].set(aspect='equal', adjustable='datalim')
        ax[j].set_xlabel(axes_labels[0])
    return fig, ax


def generate_all_plots(error_list, subtitle=None):
    # RMSE Plots
    this_stat = 'rmse'
    y_max = np.max(list(map(lambda x: x[this_stat].max().max(), error_list)))
    runs = [error_stats['name'] for error_stats in error_list]
    for hor in [15, 30, 45, 60]:
        plt.figure()
        for aresult in error_list:
            aresult[this_stat][hor].dropna().plot(linestyle='--', marker='o')
        (aresult[this_stat][hor]*np.nan).plot(color='b')
        plt.legend(runs)
        plt.ylim([0, y_max])
        plt.ylabel('RMSE (CI)')
        if subtitle is not None:
            plt.title(f'{hor} minute rmse: {subtitle}')
        else:
            plt.title(f'{hor} minute rmse')

    # Centered RMSE Plots
    y_max = None
    for hor in [15, 30, 45, 60]:
        plt.figure()
        for aresult in error_list:
            crmse = np.sqrt(
                aresult['rmse'][hor]**2
                - aresult['bias'][hor]**2)
            crmse.dropna().plot(linestyle='--', marker='o')
        (aresult['rmse'][hor]*np.nan).plot(color='b')
        plt.legend(runs)
        plt.ylim([0, y_max])
        plt.ylabel('Centered RMSE (CI)')
        if subtitle is not None:
            plt.title(f'{hor} minute centered rmse: {subtitle}')
        else:
            plt.title(f'{hor} minute centered rmse')

    # Bias plot
    this_stat = 'bias'
    y_max = np.max(list(map(
        lambda x: x[this_stat].abs().max().max(), error_list)))
    y_min = -y_max
    for hor in [15, 30, 45, 60]:
        plt.figure()
        for aresult in error_list:
            aresult[this_stat][hor].dropna().plot(linestyle='--', marker='o')
        (aresult[this_stat][hor]*np.nan).plot(color='b')
        plt.legend(runs)
        plt.ylim([y_min, y_max])
        plt.ylabel('Bias (CI)')
        if subtitle is not None:
            plt.title(f'{hor} minute bias: {subtitle}')
        else:
            plt.title(f'{hor} minute bias')

    # Correlation plot
    this_stat = 'correlation'
    y_max = np.max(list(map(lambda x: x[this_stat].max().max(), error_list)))
    y_min = np.min(list(map(lambda x: x[this_stat].min().min(), error_list)))
    for hor in [15, 30, 45, 60]:
        plt.figure()
        for aresult in error_list:
            aresult[this_stat][hor].dropna().plot(linestyle='--', marker='o')
        (aresult[this_stat][hor]*np.nan).plot(color='b')
        plt.legend(runs)
        plt.ylim([y_min, y_max])
        plt.ylabel('Correlation (unitless)')
        if subtitle is not None:
            plt.title(f'{hor} minute correlation: {subtitle}')
        else:
            plt.title(f'{hor} minute correlation')

    # Standard deviation plot
    this_stat = 'forecast_sd'
    y_max = np.max(list(map(lambda x: x[this_stat].max().max(), error_list)))
    y_max = np.max([error_list[0]['truth_sd'].max().max(), y_max])
    y_min = 0
    for hor in [15, 30, 45, 60]:
        plt.figure()
        for aresult in error_list:
            aresult[this_stat][hor].dropna().plot(linestyle='--', marker='o')
        error_list[0]['truth_sd'].dropna().plot(linestyle='--', marker='o')
        (aresult[this_stat][hor]*np.nan).plot(color='b')

        plt.legend(runs + ['truth'])
        plt.ylim([y_min, y_max])
        plt.ylabel('SD (CI)')
        if subtitle is not None:
            plt.title(f'{hor} minute standard deviation: {subtitle}')
        else:
            plt.title(f'{hor} minute standard deviation')


def generate_spread_plots(error_list, subtitle=None):
    runs = [error_stats['name'] for error_stats in error_list]

    # Cloudiness index spread plot
    this_stat = 'spread_ci'
    y_max = np.max(list(map(lambda x: x[this_stat].max().max(), error_list)))
    y_min = 0
    for hor in [15, 30, 45, 60]:
        plt.figure()
        for aresult in error_list:
            aresult[this_stat][hor].dropna().plot(
                linestyle='--', marker='o')
        (aresult[this_stat][hor]*np.nan).plot(color='b')
        plt.legend(runs + ['truth'])
        plt.ylim([y_min, y_max])
        plt.ylabel('Spread (CI)')
        if subtitle is not None:
            plt.title(f'{hor} minute CI spread: {subtitle}')
        else:
            plt.title(f'{hor} minute CI spread')

    # U spread plot
    this_stat = 'u_spread'
    y_max = np.max(list(map(lambda x: x[this_stat].max().max(), error_list)))
    y_min = 0
    for hor in [15, 30, 45, 60]:
        plt.figure()
        for aresult in error_list:
            aresult[this_stat][hor].dropna().plot(
                linestyle='--', marker='o')
        (aresult[this_stat][hor]*np.nan).plot(color='b')
        plt.legend(runs + ['truth'])
        plt.ylim([y_min, y_max])
        plt.ylabel('spread (m/s)')
        if subtitle is not None:
            plt.title(f'{hor} minute U spread: {subtitle}')
        else:
            plt.title(f'{hor} minute U spread')

    # V spread plot
    this_stat = 'v_spread'
    y_max = np.max(list(map(lambda x: x[this_stat].max().max(), error_list)))
    y_min = 0
    for hor in [15, 30, 45, 60]:
        plt.figure()
        for aresult in error_list:
            aresult[this_stat][hor].dropna().plot(
                linestyle='--', marker='o')
        (aresult[this_stat][hor]*np.nan).plot(color='b')
        plt.legend(runs + ['truth'])
        plt.ylim([y_min, y_max])
        plt.ylabel('spread (m/s)')
        if subtitle is not None:
            plt.title(f'{hor} minute V spread: {subtitle}')
        else:
            plt.title(f'{hor} minute V spread')

    # Spread vs RMSE plot
    this_stat = 'rmse'
    y_max = np.max(list(map(lambda x: x['rmse'].max().max(), error_list)))
    y_max_1 = np.max(list(map(
        lambda x: x['spread_ci'].max().max(), error_list)))
    y_max = np.max([y_max, y_max_1])
    for hor in [15, 30, 45, 60]:
        plt.figure()
        colors = ['b', 'orange', 'r', 'k', 'g', 'y']
        color_count = 0
        for aresult in error_list:
            c = colors[color_count]
            color_count += 1
            aresult['rmse'][hor].dropna().plot(marker='o',
                                               color=c)
            aresult['spread_ci'][hor].dropna().plot(
                linestyle='--', marker='*', color=c)
            used_result = aresult.copy()
        (used_result[this_stat][hor]*np.nan).plot(color='b')
        plt.legend(runs)
        plt.ylim([0, y_max])
        plt.ylabel('RMSE (CI)')
        if subtitle is not None:
            plt.title(f'{hor} minute rmse vs spread: {subtitle}')
        else:
            plt.title(f'{hor} minute rmse vs spread')
