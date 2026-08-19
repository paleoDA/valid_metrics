import matplotlib.pyplot as plt
from cartopy.crs import EqualEarth, PlateCarree
from matplotlib.gridspec import GridSpec
import xeofs as xe
import cfr


def load_recon_tas(recon_path, fields=("tas", "tas_gm"), var="tas", ens_dim="ens"):
    res = cfr.ReconRes(recon_path)
    res.load(list(fields))
    return res.da[var].mean(dim=ens_dim)


def plot_eofs(tas, n_modes=4, plot_modes=3, fit_dim="time", central_longitude=180):
    eof = xe.single.EOF(n_modes=n_modes, use_coslat=True)
    eof.fit(tas, dim=fit_dim)
    expvar = eof.explained_variance_ratio()
    components = eof.components()
    scores = eof.scores()

    proj = EqualEarth(central_longitude=central_longitude)
    kwargs = {"cmap": "RdBu_r", "transform": PlateCarree()}
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(plot_modes, 2, width_ratios=[2, 1])
    ax0 = [fig.add_subplot(gs[i, 0], projection=proj) for i in range(plot_modes)]
    ax1 = [fig.add_subplot(gs[i, 1]) for i in range(plot_modes)]

    for i, (a0, a1) in enumerate(zip(ax0, ax1)):
        mode = i + 1
        components.sel(mode=mode).plot(ax=a0, **kwargs)
        a0.coastlines(color=".5")
        a0.set_title(f"Mode {mode}, {100 * expvar[i]:3.1f}% variance")
        a0.set_xlabel("")
        scores.sel(mode=mode).plot(ax=a1)
        a1.set_xlabel("Time [years CE]")

    plt.tight_layout()
    return eof, expvar, components, scores, fig


if __name__ == "__main__":
    tas = load_recon_tas("./recons/lmr_reproduce_pda_ptype_res/")
    eof_lmr, expvar_lmr, components_lmr, scores_lmr, fig = plot_eofs(tas)
    # fig.savefig("LMRv2.1*_tas_eofs1-3.pdf", dpi=300)
