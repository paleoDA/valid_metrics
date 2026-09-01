import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cfr


def pick_top_events(years, values, min_sep_years=3, max_events=30):
    mask = years > 0
    years_f = years[mask]
    values_f = values[mask]
    order = np.argsort(values_f)[::-1]

    chosen_years = []
    chosen_vals = []

    for idx in order:
        y = years_f[idx]
        v = values_f[idx]

        if not chosen_years:
            ok = True
        else:
            diffs = np.abs(y - np.array(chosen_years))
            ok = np.all(diffs > min_sep_years)

        if ok:
            chosen_years.append(y)
            chosen_vals.append(v)
            if max_events is not None and len(chosen_years) >= max_events:
                break

    return np.array(chosen_years), np.array(chosen_vals)


def top_events_from_aod(gm_aod, min_sep_years=5, max_events=30):
    year_int = np.floor(gm_aod["time"].values).astype(int)
    year_da = xr.DataArray(
        year_int,
        coords={"time": gm_aod["time"]},
        dims="time",
        name="year",
    )
    gm_aod_annual = gm_aod.groupby(year_da).max("time")
    years = gm_aod_annual["year"].values.astype(int)
    vals = gm_aod_annual.values
    top_years, top_vals = pick_top_events(
        years,
        vals,
        min_sep_years=min_sep_years,
        max_events=max_events,
    )
    return pd.DataFrame({"year": top_years, "aod550": top_vals})


def field_sea(yr, fd, before=3, after=6):
    bl_mean = fd[str(yr - before):str(yr)].da.mean(dim="time")
    mean_field = cfr.ClimateField(bl_mean)

    sea_list = []
    for lag in range(-before, after + 1):
        year_field = fd[str(yr + lag)]
        demeaned = year_field - mean_field
        sea_list.append(demeaned.da)

    sea_da = xr.concat(sea_list, dim="time")
    sea_da = sea_da.assign_coords({"time": range(-before, after + 1)})
    return cfr.ClimateField(sea_da)


def composite_sea(eruption_years, fd, before=3, after=6):
    sea_list = []
    for yr in eruption_years:
        sea_fd = field_sea(yr, fd, before=before, after=after)
        sea_list.append(sea_fd.da)

    all_eruptions = xr.concat(sea_list, dim="eruption")
    composite_da = all_eruptions.mean(dim="eruption")
    composite_da = composite_da.assign_coords({"time": range(-before, after + 1)})
    return cfr.ClimateField(composite_da)


def bootstrap(df, fd, before=3, after=6, n_perm=1000):
    volc_yrs = df["year"].values
    all_years = fd.da.time.values

    eruption_window_years = []
    for volc_yr in volc_yrs:
        window = np.arange(volc_yr - before, volc_yr + after + 1)
        eruption_window_years.extend(window)

    eruption_window_years = np.unique(eruption_window_years)
    min_year = all_years.min() + before
    max_year = all_years.max() - after

    non_erup = []
    for yr in all_years:
        if (yr not in eruption_window_years) and (yr >= min_year) and (yr <= max_year):
            non_erup.append(yr)

    non_erup = np.array(non_erup)
    bootstrap_composites = []

    for _ in range(n_perm):
        rnd_yrs = np.random.choice(non_erup, len(volc_yrs), replace=False)
        composite_bs = composite_sea(rnd_yrs, fd, before=before, after=after)
        bootstrap_composites.append(composite_bs.da.values)

    return np.stack(bootstrap_composites, axis=0), non_erup


def bootstrap_significance(comp_fd, bootstrap_composites, ci=(2.5, 97.5)):
    ci_lo = np.percentile(bootstrap_composites, ci[0], axis=0)
    ci_up = np.percentile(bootstrap_composites, ci[1], axis=0)
    volc_comp = comp_fd.da.values
    sig_cool = volc_comp < ci_lo
    sig_warm = volc_comp > ci_up
    return ci_lo, ci_up, sig_cool, sig_warm


def difference_field(fd1, fd2, name=None):
    if fd1.da.shape != fd2.da.shape:
        raise ValueError("Climate fields must have the same shape to take a difference.")

    diff_da = fd2.da - fd1.da
    if name is not None:
        diff_da.name = name

    return cfr.ClimateField(diff_da)


def plot_lag_with_significance(
    comp_fd,
    warm,
    cool,
    lag=1,
    name="",
    levels=None,
    cbar_labels=None,
    hatch="///",
    proxydb=None,
    plot_proxydb=False,
    plot_proxydb_lgd=True,
    site_markersize=20,
):
    if levels is None:
        levels = np.linspace(-0.5, 0.5, 21)
    if cbar_labels is None:
        cbar_labels = np.linspace(-0.5, 0.5, 11)

    lag_values = comp_fd.da.time.values
    lag_matches = np.where(lag_values == lag)[0]
    if len(lag_matches) == 0:
        raise ValueError(f"Lag {lag} is not in composite lags: {lag_values}.")

    lag_idx = lag_matches[0]
    volcanic_field = cfr.ClimateField(comp_fd.da.isel(time=lag_idx))
    insig_mask = ~(warm[lag_idx, :, :] | cool[lag_idx, :, :])

    fig, ax = volcanic_field.plot(
        levels=levels,
        cbar_labels=cbar_labels,
        extend="both",
        proxydb=proxydb,
        plot_proxydb=plot_proxydb,
        plot_proxydb_lgd=plot_proxydb_lgd,
        site_markersize=site_markersize,
    )

    lat = volcanic_field.da.lat.values
    lon = volcanic_field.da.lon.values

    plt.rcParams["hatch.color"] = "gray"
    plt.rcParams["hatch.linewidth"] = 0.5
    ax.contourf(
        lon,
        lat,
        insig_mask.astype(float),
        levels=[0.5, 1.5],
        colors="none",
        hatches=[hatch],
        transform=ccrs.PlateCarree(),
        alpha=0,
    )

    ax.set_title(f"{name} Composite, Year {lag}")
    cbar_ax = fig.axes[-1]
    cbar_ax.set_title(r"$T$", pad=15)

    return fig, ax


def plot_field_sea_sig(
    events_df,
    fd,
    lag=1,
    before=3,
    after=6,
    n_perm=1000,
    ci=(2.5, 97.5),
    name="",
    levels=None,
    cbar_labels=None,
    hatch="///",
    proxydb=None,
    plot_proxydb=False,
    plot_proxydb_lgd=True,
    site_markersize=20,
):
    eruption_years = events_df["year"].values
    comp_fd = composite_sea(eruption_years, fd, before=before, after=after)
    bootstrap_composites, non_erup = bootstrap(
        events_df,
        fd,
        before=before,
        after=after,
        n_perm=n_perm,
    )
    ci_lo, ci_up, sig_cool, sig_warm = bootstrap_significance(
        comp_fd,
        bootstrap_composites,
        ci=ci,
    )
    fig, ax = plot_lag_with_significance(
        comp_fd,
        sig_warm,
        sig_cool,
        lag=lag,
        name=name,
        levels=levels,
        cbar_labels=cbar_labels,
        hatch=hatch,
        proxydb=proxydb,
        plot_proxydb=plot_proxydb,
        plot_proxydb_lgd=plot_proxydb_lgd,
        site_markersize=site_markersize,
    )

    return {
        "composite": comp_fd,
        "bootstrap_composites": bootstrap_composites,
        "non_erup": non_erup,
        "ci_lo": ci_lo,
        "ci_up": ci_up,
        "sig_cool": sig_cool,
        "sig_warm": sig_warm,
        "fig": fig,
        "ax": ax,
    }


def plot_comp_with_null(
    comp_fd,
    warm,
    cool,
    name="",
    levels=None,
    cbar_labels=None,
    proxydb=None,
    plot_proxydb=False,
    plot_proxydb_lgd=True,
    site_markersize=20,
):
    if levels is None:
        levels = np.linspace(-0.5, 0.5, 21)
    if cbar_labels is None:
        cbar_labels = np.linspace(-0.5, 0.5, 11)

    figs = []
    for lag_idx in range(len(comp_fd.da.time)):
        volcanic_field = cfr.ClimateField(comp_fd.da.isel(time=lag_idx))
        sig_cooling_mask = cool[lag_idx, :, :]
        sig_warming_mask = warm[lag_idx, :, :]
        insig_mask = ~(sig_cooling_mask | sig_warming_mask)

        fig, ax = volcanic_field.plot(
            levels=levels,
            cbar_labels=cbar_labels,
            extend="both",
            proxydb=proxydb,
            plot_proxydb=plot_proxydb,
            plot_proxydb_lgd=plot_proxydb_lgd,
            site_markersize=site_markersize,
        )

        lat = volcanic_field.da.lat.values
        lon = volcanic_field.da.lon.values

        plt.rcParams["hatch.color"] = "gray"
        plt.rcParams["hatch.linewidth"] = 0.5
        ax.contourf(
            lon,
            lat,
            insig_mask.astype(float),
            levels=[0.5, 1.5],
            colors="none",
            hatches=[".."],
            transform=ccrs.PlateCarree(),
            alpha=0,
        )

        lag = comp_fd.da.time.values[lag_idx]
        plt.title(f"{name} Composite, Year {lag}")
        cbar_ax = fig.axes[-1]
        cbar_ax.set_title(r"$T$", pad=15)
        figs.append(fig)

    return figs


def sea_composite_ensemble(ens_ts, eruption_years, window=(-3, 6)):
    start_lag, end_lag = window
    window_length = end_lag - start_lag + 1
    lags = np.arange(start_lag, end_lag + 1)
    composites = []

    for year in eruption_years:
        window_mask = (ens_ts.time >= year + start_lag) & (ens_ts.time <= year + end_lag)
        baseline_mask = (ens_ts.time >= year - 3) & (ens_ts.time <= year - 1)

        if np.sum(window_mask) == window_length and np.sum(baseline_mask) == 3:
            window_data = ens_ts.value[window_mask, :]
            baseline = np.mean(ens_ts.value[baseline_mask, :], axis=0)
            window_anomaly = window_data - baseline[np.newaxis, :]
            composites.append(window_anomaly)

    composites = np.array(composites)
    composite_mean = np.mean(composites, axis=0)

    return cfr.EnsTS(
        time=lags,
        value=composite_mean,
        value_name=ens_ts.value_name,
    )


def difference_ensemble_sea(ens_ts_1, ens_ts_2, name=None):
    if not np.array_equal(ens_ts_1.time, ens_ts_2.time):
        raise ValueError("SEA composites must use the same lags to take a difference.")

    if np.shape(ens_ts_1.value) != np.shape(ens_ts_2.value):
        raise ValueError("SEA composites must have the same ensemble shape to take a difference.")

    if name is None:
        name = f"{ens_ts_2.value_name} - {ens_ts_1.value_name}"

    return cfr.EnsTS(
        time=ens_ts_1.time,
        value=ens_ts_2.value - ens_ts_1.value,
        value_name=name,
    )


def plot_ensemble_sea(ens_series, labels, colors, title="SEA", ylabel="Temperature anomaly (deg C)"):
    fig, ax = plt.subplots(figsize=(12, 6))

    for ens_ts, label, color in zip(ens_series, labels, colors):
        mean = ens_ts.mean
        std = ens_ts.std
        lags = ens_ts.time
        ax.plot(lags, mean, color=color, linewidth=2, marker="o", label=label)
        ax.fill_between(lags, mean - std, mean + std, color=color, alpha=0.2)

    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.axvline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Years relative to eruption")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig, ax
