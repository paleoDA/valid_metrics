import numpy as np


def plot_volcanic_year(
    fd,
    pdb,
    year,
    name="",
    other_fd=None,
    levels=None,
    cbar_labels=None,
    cbar_title=None,
    plot_proxydb=True,
    site_markersize=80,
    figsize=(12, 8),
):
    from cfr.proxy import ProxyDatabase

    filtered_pdb = ProxyDatabase()
    for _, pobj in pdb.records.items():
        if year in pobj.time:
            filtered_pdb += pobj
    filtered_pdb.refresh()

    if other_fd is None:
        plot_fd = fd[str(year)]
        title = f"{name} Year {year} CE (n={filtered_pdb.nrec} proxies)"
        if levels is None:
            levels = np.linspace(-1.5, 1.5, 21)
        if cbar_labels is None:
            cbar_labels = np.linspace(-1.5, 1.5, 11)
        if cbar_title is None:
            cbar_title = r"$T$"
    else:
        plot_fd = other_fd[str(year)] - fd[str(year)]
        title = f"{name} Year {year} CE (Delta n={filtered_pdb.nrec} proxies)"
        if levels is None:
            levels = np.linspace(-1.5, 1.5, 21)
        if cbar_labels is None:
            cbar_labels = np.linspace(-1.5, 1.5, 11)
        if cbar_title is None:
            cbar_title = r"$\Delta T$"

    fig, ax = plot_fd.plot(
        title=title,
        plot_proxydb=plot_proxydb,
        proxydb=filtered_pdb,
        plot_proxydb_lgd=True,
        proxydb_lgd_kws={
            "bbox_to_anchor": (1.2, 1),
            "loc": "upper left",
            "frameon": False,
            "ncol": 1,
        },
        site_markersize=site_markersize,
        levels=levels,
        cbar_labels=cbar_labels,
        extend="both",
        figsize=figsize,
    )
    cbar_ax = fig.axes[-1]
    cbar_ax.set_title(cbar_title, pad=15)

    return filtered_pdb, fig, ax
