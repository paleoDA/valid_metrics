import numpy as np
import cfr
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection

## Integrated Tukey depth functions

def depth_vectorized(g, fmat):
    """
    Compute univariate Tukey depth at each gridpoint.
    
    Parameters
    ----------
    g : ndarray, shape (n_gridpoints,)
        Single ensemble member to evaluate
    fmat : ndarray, shape (n_gridpoints, n_ensemble)
        Reference ensemble
    
    Returns
    -------
    ndarray, shape (n_gridpoints,)
        Depth at each gridpoint
    """
    n_ensemble = fmat.shape[1]
    
    # Broadcasting: compare g to all ensemble members
    signs = np.sign(g[:, None] - fmat)
    diff = np.abs(np.sum(signs, axis=1))
    depth_values = 1 - (diff / n_ensemble)
    
    return depth_values


def int_depth_vectorized(gmat, fmat):
    """
    Compute integrated Tukey depth for each ensemble member in gmat.
    
    Parameters
    ----------
    gmat : ndarray, shape (n_gridpoints, n_ensemble_g)
        Ensemble to evaluate
    fmat : ndarray, shape (n_gridpoints, n_ensemble_f)
        Reference ensemble
    
    Returns
    -------
    ndarray, shape (n_ensemble_g,)
        Integrated depth for each member of gmat
    """
    n_ensemble_g = gmat.shape[1]
    
    integrated_depths = np.array([
        np.mean(depth_vectorized(gmat[:, j], fmat))
        for j in range(n_ensemble_g)
    ])
    
    return integrated_depths


def reshape_for_depth(data_array, year_idx):
    """
    Extract and reshape data for a single year.
    
    Parameters
    ----------
    data_array : xarray.DataArray
        Shape (time, ens, lat, lon)
    year_idx : int
        Which year to extract
    
    Returns
    -------
    ndarray, shape (n_gridpoints, n_ensemble)
    """
    year_data = data_array[year_idx, :, :, :]
    n_ens = year_data.shape[0]
    n_gridpoints = year_data.shape[1] * year_data.shape[2]
    
    year_reshaped = year_data.values.reshape(n_ens, n_gridpoints)
    return year_reshaped.T


def _drop_duplicate_cyclic_longitude(data_array_1, data_array_2):
    """Drop a shared duplicate longitude endpoint such as 0 and 360 degrees.

    A cyclic latitude-longitude grid sometimes includes both endpoints even
    though they represent the same meridian. Keeping both would give that
    meridian twice the weight of every other longitude in the integrated
    depth. The endpoint is removed only when both arrays have matching
    longitude coordinates and their first and last coordinates differ by a
    full 360-degree cycle.
    """
    if 'lon' not in data_array_1.dims or 'lon' not in data_array_2.dims:
        return data_array_1, data_array_2, False

    lon_1 = np.asarray(data_array_1.coords['lon'].values)
    lon_2 = np.asarray(data_array_2.coords['lon'].values)

    if lon_1.size < 2 or lon_2.size < 2:
        return data_array_1, data_array_2, False

    matching_longitudes = (
        lon_1.shape == lon_2.shape
        and np.allclose(lon_1, lon_2, equal_nan=False)
    )
    cyclic_endpoints = (
        np.isfinite(lon_1[0])
        and np.isfinite(lon_1[-1])
        and np.isclose(abs(lon_1[-1] - lon_1[0]), 360.0)
    )

    if not (matching_longitudes and cyclic_endpoints):
        return data_array_1, data_array_2, False

    return (
        data_array_1.isel(lon=slice(None, -1)),
        data_array_2.isel(lon=slice(None, -1)),
        True,
    )


def _validate_finite_matrix(matrix, input_name, year):
    """Reject missing or infinite field values before computing depth."""
    finite = np.isfinite(matrix)
    if np.all(finite):
        return

    invalid_count = int(matrix.size - np.count_nonzero(finite))
    raise ValueError(
        f"{input_name} contains {invalid_count} non-finite value(s) in "
        f"year {year}. KD was not computed because Harris's integrated "
        "depth does not define missing-value handling."
    )


def _handle_nonfinite_matrices(fmat, gmat, year, missing):
    """Apply the requested missing-value policy to a pair of yearly fields."""
    if missing == 'raise':
        _validate_finite_matrix(fmat, 'data_array_1', year)
        _validate_finite_matrix(gmat, 'data_array_2', year)
        return fmat, gmat

    common_valid = (
        np.all(np.isfinite(fmat), axis=1)
        & np.all(np.isfinite(gmat), axis=1)
    )
    n_gridpoints_used = int(np.count_nonzero(common_valid))
    if n_gridpoints_used == 0:
        raise ValueError(
            f"No grid points contain finite values for every ensemble "
            f"member in both inputs in year {year}. KD is undefined."
        )

    return fmat[common_valid, :], gmat[common_valid, :]

## Main KStat Function


def ks_cdf(x, n=20):
    """
    Calculate the CDF of the Kolmogorov distribution.
    
    Parameters
    ----------
    x : float
        A positive number
    n : int, optional
        Number of terms to include in the series. Default is 20.
    
    Returns
    -------
    float
        Probability P(K <= x) under Kolmogorov distribution
    
    Notes
    -----
    This implements the series expansion:
    P(K <= x) = 1 - 2 * sum_{k=1}^{n} (-1)^(k-1) * exp(-2k²x²)
    """
    if x < 0.05:
        return 0.0
    
    # Compute the alternating sum
    terms = np.array([
        ((-1)**(k-1)) * np.exp(-2 * (k**2) * (x**2))
        for k in range(1, n+1)
    ])
    
    return 1 - 2 * np.sum(terms)


def kd_critical_value(n_f, n_g, alpha=0.05, n_terms=20):
    """
    Pointwise asymptotic KD critical value for two ensemble sizes.

    This is the inverse of the same Kolmogorov distribution approximation used
    by kstat, scaled by sqrt((n_f * n_g) / (n_f + n_g)).
    """
    if n_f <= 0 or n_g <= 0:
        raise ValueError("Ensemble sizes must be positive.")

    target = 1 - alpha
    lo = 0.0
    hi = 1.0

    while ks_cdf(hi, n=n_terms) < target:
        hi *= 2

    for _ in range(80):
        mid = (lo + hi) / 2
        if ks_cdf(mid, n=n_terms) < target:
            lo = mid
        else:
            hi = mid

    rate = np.sqrt((n_f * n_g) / (n_f + n_g))
    return hi / rate


def kstat(fmat, gmat):
    """
    Test if functions in fmat and gmat are equal in distribution using KD statistic.
    
    This is the core Kolmogorov Depth (KD) test statistic from Harris et al. (2021).
    
    Parameters
    ----------
    fmat : ndarray, shape (n_gridpoints, n_ensemble_f)
        First ensemble (e.g., LMR reconstruction)
    gmat : ndarray, shape (n_gridpoints, n_ensemble_g)
        Second ensemble (e.g., PReSto reconstruction)
    
    Returns
    -------
    tuple (kd_statistic, p_value)
        kd_statistic : float
            The KD test statistic
        p_value : float
            P-value under null hypothesis of equal distributions
    
    Notes
    -----
    The test is symmetric and tests H0: P = Q vs Ha: P ≠ Q
    where P and Q are the distributions of fmat and gmat.
    """
    # Step 1: Compute integrated depths
    # ff.xd: depth of fmat members relative to fmat
    ff_xd = int_depth_vectorized(fmat, fmat)
    
    # fg.xd: depth of fmat members relative to gmat  
    fg_xd = int_depth_vectorized(fmat, gmat)
    
    # gg.xd: depth of gmat members relative to gmat
    gg_xd = int_depth_vectorized(gmat, gmat)
    
    # gf.xd: depth of gmat members relative to fmat
    gf_xd = int_depth_vectorized(gmat, fmat)
    
    # Step 2: Compute empirical CDFs
    # For each depth value in ff.xd, what proportion of ff.xd values are <= it?
    ff_cdf = np.array([np.mean(ff_xd <= y) for y in ff_xd])
    
    # For each depth value in ff.xd, what proportion of gf.xd values are <= it?
    gf_cdf = np.array([np.mean(gf_xd <= y) for y in ff_xd])
    
    # For each depth value in gg.xd, what proportion of fg.xd values are <= it?
    fg_cdf = np.array([np.mean(fg_xd <= y) for y in gg_xd])
    
    # For each depth value in gg.xd, what proportion of gg.xd values are <= it?
    gg_cdf = np.array([np.mean(gg_xd <= y) for y in gg_xd])
    
    # Step 3: Compute KD statistic components
    n_f = fmat.shape[1]
    n_g = gmat.shape[1]
    rate = np.sqrt((n_g * n_f) / (n_g + n_f))
    
    # Maximum deviation using fmat as reference
    ksf = np.max(np.abs(ff_cdf - gf_cdf))
    
    # Maximum deviation using gmat as reference
    ksg = np.max(np.abs(gg_cdf - fg_cdf))
    
    # Overall KD statistic (symmetric)
    kd = max(ksf, ksg)
    
    # Step 4: Compute p-value using Kolmogorov distribution
    p_value = 1 - ks_cdf(rate * kd)
    
    return kd, p_value


def kstat_perm(fmat, gmat, perms=500, random_state=None):
    """
    Test if functions in fmat and gmat are equal in distribution using 
    permutation-based p-value.
    
    Parameters
    ----------
    fmat : ndarray, shape (n_gridpoints, n_ensemble_f)
        First ensemble
    gmat : ndarray, shape (n_gridpoints, n_ensemble_g)
        Second ensemble
    perms : int, optional
        Number of permutations for permutation test. Default is 500.
    random_state : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    tuple (kd_statistic, p_value_perm)
        kd_statistic : float
            The KD test statistic
        p_value_perm : float
            Permutation-based p-value
    
    Notes
    -----
    This is more computationally expensive but doesn't rely on 
    asymptotic Kolmogorov distribution approximation.
    Useful for small samples or when asymptotic assumptions may not hold.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Compute observed KD statistic
    kd_observed, _ = kstat(fmat, gmat)
    
    # Combine both ensembles
    hmat = np.column_stack([fmat, gmat])
    n_total = hmat.shape[1]
    n_f = fmat.shape[1]
    
    # Generate permutation distribution
    kd_dist = np.zeros(perms)
    
    for i in range(perms):
        # Randomly permute columns
        perm_idx = np.random.permutation(n_total)
        hstar = hmat[:, perm_idx]
        
        # Split into two groups of original sizes
        fstar = hstar[:, :n_f]
        gstar = hstar[:, n_f:]
        
        # Compute KD statistic for permuted data
        kd_dist[i], _ = kstat(fstar, gstar)
    
    # Compute permutation p-value
    p_value_perm = np.mean(kd_dist > kd_observed)
    
    return kd_observed, p_value_perm

## Full Algorithm
def compute_kd_timeseries(
    data_array_1,
    data_array_2,
    use_permutation=False,
    perms=500,
    missing='raise',
):
    """
    Compute time-evolving KD statistic comparing two reconstructions.
    
    Parameters
    ----------
    data_array_1 : xarray.DataArray
        First reconstruction (e.g., LMR), shape (time, ens, lat, lon)
    data_array_2 : xarray.DataArray
        Second reconstruction (e.g., PReSto), shape (time, ens, lat, lon)
    use_permutation : bool, optional
        If True, use permutation test (slower). Default False uses asymptotic.
    perms : int, optional
        Number of permutations if use_permutation=True
    missing : {'raise', 'drop'}, optional
        How to handle non-finite values. ``'raise'`` stops with an informative
        error. ``'drop'`` removes, for each year, any grid point that is not
        finite in every member of both ensembles. Default is ``'raise'``.
    
    Returns
    -------
    dict with keys:
        'kd_stats': array (n_years,) - KD statistic per year
        'p_values': array (n_years,) - p-value per year
        'years': array (n_years,) - year labels from data
    """
    if missing not in {'raise', 'drop'}:
        raise ValueError("missing must be either 'raise' or 'drop'.")

    data_array_1, data_array_2, _ = (
        _drop_duplicate_cyclic_longitude(data_array_1, data_array_2)
    )

    n_years = data_array_1.shape[0]
    
    kd_stats = np.zeros(n_years)
    p_values = np.zeros(n_years)
    
    for year_idx in range(n_years):
        year = data_array_1.coords['time'].values[year_idx]

        if year_idx % 100 == 0:
            print(
                f"Processing year {year_idx}/{n_years} "
                f"(year {year})..."
            )
        
        # Prepare data for this year
        fmat = reshape_for_depth(data_array_1, year_idx)
        gmat = reshape_for_depth(data_array_2, year_idx)
        fmat, gmat = _handle_nonfinite_matrices(
            fmat,
            gmat,
            year,
            missing,
        )
        
        # Compute KD statistic
        if use_permutation:
            kd_stats[year_idx], p_values[year_idx] = kstat_perm(fmat, gmat, perms=perms)
        else:
            kd_stats[year_idx], p_values[year_idx] = kstat(fmat, gmat)
    
    return {
        'kd_stats': kd_stats,
        'p_values': p_values,
        'years': data_array_1.coords['time'].values,
    }

## Compute and Plot

def _mean_ts_values(ts):
    """Return a 1D mean time series from cfr EnsTS or xarray ensemble series."""
    if isinstance(ts, xr.DataArray):
        if 'ens' in ts.dims:
            return ts.mean(dim='ens').values
        return ts.values

    mean = getattr(ts, 'mean', None)
    if mean is not None and not callable(mean):
        return np.asarray(mean)

    raise TypeError(
        "Expected a cfr EnsTS-like object or an xarray.DataArray for ts."
    )


def compute_and_plot_kd(
    fd1,
    fd2,
    ts1,
    ts2,
    name1,
    name2,
    region=None,
    highlight_threshold=0.9,
    missing='raise',
):

    print(f"Computing KD statistic for {name1} vs {name2}...")
    results = compute_kd_timeseries(
        fd1,
        fd2,
        use_permutation=False,  # Use asymptotic approximation
        missing=missing,
    )

    print(f"\nCompleted {len(results['kd_stats'])} years")
    print(
        f"KD range: "
        f"[{results['kd_stats'].min():.4f}, "
        f"{results['kd_stats'].max():.4f}]"
    )
    print(
        f"Raw p-value range: "
        f"[{results['p_values'].min():.4e}, "
        f"{results['p_values'].max():.4e}]"
    )

    # ========================================
    # Step 2: Benjamini-Yekutieli FDR correction
    # ========================================

    alpha = 0.05
    pvals = results['p_values']

    rejected, pvals_adjusted = fdrcorrection(
        pvals,
        alpha=alpha,
        method='negcorr'  # BY correction for arbitrary dependence
    )

    n_tests = len(pvals)
    n_significant = np.sum(rejected)
    sig_years_fdr = results['years'][rejected]

    print("\n=== Statistical Summary (Benjamini-Yekutieli FDR) ===")
    print(
        f"Significant years (FDR alpha={alpha}): "
        f"{n_significant}/{n_tests} "
        f"({100 * n_significant / n_tests:.1f}%)"
    )
    print(f"Median KD statistic: {np.median(results['kd_stats']):.4f}")
    print(f"FDR-significant years: {len(sig_years_fdr)} years")
    print(
        f"FDR-adjusted p-value range: "
        f"[{pvals_adjusted.min():.4e}, "
        f"{pvals_adjusted.max():.4e}]"
    )

    # ========================================
    # Step 3: Absolute GMST median difference
    # ========================================

    mean_diff = np.abs(_mean_ts_values(ts1) - _mean_ts_values(ts2))

    # ========================================
    # Step 3.5: Table sorted by GMST difference
    # ========================================

    sorted_idx = np.argsort(mean_diff)[::-1]

    kd_gmst_table = pd.DataFrame({
        'year': results['years'][sorted_idx],
        'kd_stat': results['kd_stats'][sorted_idx],
        'abs_gmst_diff': mean_diff[sorted_idx],
        'p_value_raw': pvals[sorted_idx],
        'p_value_fdr': pvals_adjusted[sorted_idx],
        'significant_fdr': rejected[sorted_idx]
    })

    # ========================================
    # Step 4: Two-panel visualization
    # ========================================

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 10), sharex=True
    )

    years = results['years']
    n_f = fd1.sizes['ens']
    n_g = fd2.sizes['ens']
    kd_alpha = 0.05
    kd_threshold = kd_critical_value(n_f, n_g, alpha=kd_alpha)

    highlight_mask = results['kd_stats'] > highlight_threshold
    highlight_years = years[highlight_mask]
    highlight_spans = []

    if np.any(highlight_mask):
        diff = np.diff(
            np.concatenate(([False], highlight_mask, [False])).astype(int)
        )
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for start, end in zip(starts, ends):
            span = (years[start] - 0.5, years[end - 1] + 0.5)
            highlight_spans.append(span)
            ax1.axvspan(*span, color='lightblue', alpha=0.6)

    # Panel 1: KD statistic
    ax1.scatter(
        years,
        results['kd_stats'],
        alpha=0.5,
        s=10,
        c='black'
    )

    z = np.polyfit(years, results['kd_stats'], 1)
    trend = np.poly1d(z)
    ax1.plot(
        years,
        trend(years),
        "r-",
        linewidth=2,
        label='Trend'
    )

    ax1.set_ylabel('KD Statistic', fontsize=12)

    if region is not None:
        ax1.set_title(
            f'Time Evolution of KD Statistic {region}: '
            f'{name1} vs {name2}',
            fontsize=14,
            fontweight='bold'
        )
    else:
        ax1.set_title(
            f'Time Evolution of KD Statistic: {name1} vs {name2}',
            fontsize=14,
            fontweight='bold'
        )

    ax1.grid(True, alpha=0.3)

    # Pointwise, uncorrected alpha threshold for the loaded ensemble sizes
    ax1.axhline(
        y=kd_threshold,
        color='#000080',
        linestyle='--',
        label=(
            f'KD={kd_threshold:.2f} '
            f'(alpha={kd_alpha}, n={n_f}/{n_g})'
        )
    )
    ax1.legend()

    # Panel 2: Absolute GMST difference
    for span in highlight_spans:
        ax2.axvspan(*span, color='lightblue', alpha=0.6)

    ax2.plot(
        years,
        mean_diff,
        linewidth=1,
        color='black'
    )

    ax2.set_ylabel(
        r'|$\Delta$ GMST|' + f' ({name1} - {name2})',
        fontsize=12
    )
    ax2.set_xlabel('Year', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Store raw and FDR-corrected results
    results['p_values_adjusted'] = pvals_adjusted
    results['rejected_fdr'] = rejected
    results['sig_years_fdr'] = sig_years_fdr
    results['mean_diff'] = mean_diff
    results['kd_gmst_table'] = kd_gmst_table
    results['highlight_threshold'] = highlight_threshold
    results['highlight_years'] = highlight_years

    return results

import numpy as np

## Weighted mean for ensembles 
def weighted_mean_ts(da):
    """
    Calculate area-weighted spatial mean for ensemble data.
    Returns (time, ens) for KD analysis.
    
    Parameters:
    -----------
    da : xarray.DataArray
        DataArray with (time, ens, lat, lon) dimensions
    
    Returns:
    --------
    xarray.DataArray : Shape (time, ens) - suitable for KD analysis
    """
    # Calculate weights based on cosine of latitude
    weights = np.cos(np.deg2rad(da.lat))
    weights.name = "weights"
    
    # Weight the data and take mean over spatial dimensions only
    da_weighted = da.weighted(weights)
    
    # Mean over lat and lon, keeping time and ens
    # Result will be (time, ens)
    return da_weighted.mean(dim=['lat', 'lon'])


def compute_and_plot_kd_reg(
    fd1,
    fd2,
    ts1,
    ts2,
    name1,
    name2,
    region=None,
    highlight_threshold=0.9,
    missing='raise',
):
    """
    Regional KD wrapper matching the final-project notebook.

    Parameters are the same as compute_and_plot_kd, with fd1/fd2 usually
    pre-sliced by latitude and ts1/ts2 produced by weighted_mean_ts.
    """
    results = compute_and_plot_kd(
        fd1,
        fd2,
        ts1,
        ts2,
        name1,
        name2,
        region=region,
        highlight_threshold=highlight_threshold,
        missing=missing,
    )

    high_kd_mask = results['kd_stats'] > highlight_threshold
    n_high_kd = np.sum(high_kd_mask)
    total_years = len(results['kd_stats'])
    print(
        f"Years with KD > {highlight_threshold}: "
        f"{n_high_kd}/{total_years} "
        f"({100 * n_high_kd / total_years:.1f}%)"
    )

    return results
