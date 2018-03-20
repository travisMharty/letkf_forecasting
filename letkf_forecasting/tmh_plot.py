import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


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


def subplots(data, x, y, subplot_titles, axes_label, sup_title=None,
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
    nc = 10
    bounds = np.linspace(vmin, vmax, nc)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    suptitle_x = .5*(1 - fraction)
    nrows, ncols = 1, len(data)
    dy, dx = x.shape
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
    ax[0].set_ylabel(axes_label)
    for j in range(ncols):
        ax[j].set(aspect='equal', adjustable='datalim')
        ax[j].set_xlabel(axes_label)
    return fig, ax
