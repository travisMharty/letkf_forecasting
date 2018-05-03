import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import letkf_forecasting.analyse_results as ar


def of_vectors_plot(clouds0_8b, clouds1_8b, p0_good, p1_good, u, v,
                    time0, time1, adjust):
    adjust = 1.1
    nrows, ncols = 1, 2
    dy, dx = clouds0_8b.shape
    # figsize = plt.figaspect(float(dy * nrows) /
    #                         ((1 + fraction + pad)*float(dx * ncols)))
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

    # plt.colorbar(im, ax=ax.tolist(), pad=pad, fraction=fraction)
    # qk = plt.quiverkey(qu, .5, 0.92, 20, r'$20 \frac{m}{s}$', labelpos='E',
    #                    coordinates='figure')
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
    # qu = ax[0].quiver(p0_good[:, 0], p0_good[:, 1], u, v)
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
    # qu = ax[0].quiver(p0_good[:, 0], p0_good[:, 1], u, v)
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
        cmap='Blues', vmin=0, vmax=1):
    if vmin is None or vmax is None:
        vmax = ensemble.max()
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
        #delete
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
    # ax[0, j + 1].set_title('Mean')
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
    plt.colorbar(im, ax=ax.ravel().tolist(), pad=pad, fraction=fraction)
    # for j in range(len(other_titles)):
    #     ax_pos = ax[0, j].axis()
    #     print(ax_pos)
    #     rec = patches.Rectangle((ax_pos[0] + 1, ax_pos[2] + 2),
    #                             ax_pos[1] - ax_pos[0] - 4,
    #                             ax_pos[3] - ax_pos[2] + 65,
    #                             fill=False, lw=lw, color='k', capstyle='butt')
    #     rec = ax[0, j].add_patch(rec)
    #     rec.set_clip_on(False)
    # if j is None:
    #     j=0
    # ax_pos = ax[0, j + 1].axis()
    # rec = patches.Rectangle((ax_pos[0] + 1, ax_pos[2] + 2),
    #                         ax_pos[1] - ax_pos[0] - 4,
    #                         ax_pos[3] - ax_pos[2] + 65,
    #                         fill=False, lw=lw, color='k')
    # rec = ax[0, j + 1].add_patch(rec)
    # rec.set_clip_on(False)

    return fig, ax


def ensemble_stamps(
        others, other_titles, ensemble, nrows, ncols, domain_shape, adjust,
        cmap='Blues', vmin=0, vmax=1):
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
        #delete
        ax[0, j].set_title(other_titles[j])
        ax[0, j].axis('off')
    try:
        j
    except Exception:
        j=-1
    im = ax[0, j + 1].pcolormesh(
        ensemble.mean(axis=1).reshape(domain_shape),
        cmap=cmap,
        norm=norm)
    # ax[0, j + 1].set_title('Mean')
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
    plt.colorbar(im, ax=ax.ravel().tolist(), pad=pad, fraction=fraction)
    # for j in range(len(other_titles)):
    #     ax_pos = ax[0, j].axis()
    #     print(ax_pos)
    #     rec = patches.Rectangle((ax_pos[0] + 1, ax_pos[2] + 2),
    #                             ax_pos[1] - ax_pos[0] - 4,
    #                             ax_pos[3] - ax_pos[2] + 65,
    #                             fill=False, lw=lw, color='k', capstyle='butt')
    #     rec = ax[0, j].add_patch(rec)
    #     rec.set_clip_on(False)
    # if j is None:
    #     j=0
    # ax_pos = ax[0, j + 1].axis()
    # rec = patches.Rectangle((ax_pos[0] + 1, ax_pos[2] + 2),
    #                         ax_pos[1] - ax_pos[0] - 4,
    #                         ax_pos[3] - ax_pos[2] + 65,
    #                         fill=False, lw=lw, color='k')
    # rec = ax[0, j + 1].add_patch(rec)
    # rec.set_clip_on(False)

    return fig, ax


def subplots(data, x, y, subplot_titles, axes_labels, sup_title=None,
             cb_label=None, cmap='Blues', vmax=None, adjust=1):
    """
    data and subplot_titles are lists of the same length
    """
    fraction = 0.10
    pad = .02
    # cmap = 'Blues'
    if vmax is None:
        vmin = 0
        vmax = 1
    else:
        vmin = -vmax
    nc = 11
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
