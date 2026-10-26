"""
event_study_utils.py

Shared market-model event-study engine, factored out of `31_event_study.ipynb`
so that `31_0_event_study_assumptions_v2.ipynb` (and any future window/anchor
variant) can reuse the exact same arithmetic instead of re-deriving it.

Core design (unchanged from `31_event_study.ipynb`):
    - OLS market model fit on an estimation window [-210, -11] trading days
      relative to the event reference date.
    - MIN_EST_OBS = 50 trading days minimum for a usable fit.
    - Event-window abnormal returns (AR) = firm return - (alpha + beta * market return).
    - CAR = sum of AR over the event window.
    - SCAR = CAR standardized by its forecast-error variance (Patell/BMP-style):
          forecast_factor = L + L**2/n_est + (sum(mkt_evt) - L*x_bar)**2/sxx
          scar = car / (resid_std * sqrt(forecast_factor))

`anchor` controls how a reference date that is not itself a trading day maps
onto the trading calendar before offsets are applied:
    - 'ffill' (legacy/default): previous-or-equal trading day.
    - 'bfill': next-or-equal trading day.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------
# Shared design constants (must match 31_event_study.ipynb exactly)
# ---------------------------------------------------------------------
ESTIMATION_START = -210
ESTIMATION_END = -11
MIN_EST_OBS = 50  # MacKinlay (1997) minimum for reliable market-model estimates


def get_trading_day_offset(date_index, ref_date, offset, anchor='ffill'):
    """
    Map ref_date onto the trading calendar `date_index` using `anchor`
    ('ffill' = previous-or-equal trading day, 'bfill' = next-or-equal trading
    day), then step `offset` trading days from that anchor. Returns None if
    ref_date cannot be anchored or the offset falls outside the calendar.
    """
    if anchor not in ('ffill', 'bfill'):
        raise ValueError(f"anchor must be 'ffill' or 'bfill', got {anchor!r}")
    pos = date_index.get_indexer([ref_date], method=anchor)[0]
    if pos < 0:
        return None
    target = pos + offset
    if target < 0 or target >= len(date_index):
        return None
    return date_index[target]


def estimate_market_model(firm_rets, market_rets, est_start, est_end):
    """
    OLS market-model fit on [est_start, est_end] (calendar dates, inclusive).
    Mirrors 31_event_study.ipynb's `estimate_market_model` line-for-line so
    that downstream CAR/SCAR arithmetic reproduces it to floating-point
    precision. Returns None if fewer than MIN_EST_OBS valid paired
    observations are available.
    """
    mask = (firm_rets.index >= est_start) & (firm_rets.index <= est_end)
    y = firm_rets.loc[mask].dropna()
    x = market_rets.reindex(y.index).dropna()
    y, x = y.align(x, join='inner')
    if len(y) < MIN_EST_OBS:
        return None
    slope, intercept, *_ = stats.linregress(x, y)
    resid = y - (intercept + slope * x)
    n_est = len(y)
    resid_std = np.sqrt(np.sum(resid ** 2) / (n_est - 2))  # T-2: alpha, beta estimated
    x_bar = x.mean()
    sxx = np.sum((x - x_bar) ** 2)
    return intercept, slope, resid_std, n_est, x_bar, sxx


def event_window_stats(firm_rets, market_rets, trading_cal, ref_date, a, b,
                        anchor='ffill',
                        est_start=ESTIMATION_START, est_end=ESTIMATION_END):
    """
    Full market-model event-study computation for one event, for an arbitrary
    event window [a, b] (trading-day offsets from the anchored ref_date).

    Parameters
    ----------
    firm_rets, market_rets : pd.Series
        Log-return series indexed by trading date (NaNs allowed; dropped internally).
    trading_cal : pd.DatetimeIndex
        The trading calendar to anchor/offset against (typically the market
        return series' own index).
    ref_date : Timestamp
        The nominal event date (may or may not itself be a trading day).
    a, b : int
        Event-window trading-day offsets (inclusive), e.g. (-1, 5).
    anchor : {'ffill', 'bfill'}
        How to map ref_date onto the trading calendar before applying offsets.
    est_start, est_end : int
        Estimation-window trading-day offsets (inclusive), default [-210, -11].

    Returns
    -------
    dict or None
        None if the market model is not estimable (fewer than MIN_EST_OBS
        estimation observations) OR the event window is not fully observed
        (any day in [a, b] missing after reindexing) -- these are hard
        "not computable" conditions, matching the original notebook's
        `if model is None: continue` / `if len(ar_series) != n_days: continue`.

        On success, returns:
            alpha, beta, resid_std, n_est : market-model fit
            offsets : list of int, the day offsets a..b
            ar : np.ndarray, abnormal returns aligned to `offsets` (ar[0] is
                 offset `a`, NOT a relabeled position -- avoids the ar_day_i
                 off-by-one footgun in event_study_results_v3.parquet)
            car : float, sum of ar
            scar : float or nan
                NaN only when the *forecast-variance* denominator is
                degenerate (sxx <= 0 or resid_std <= 0) -- distinct from the
                hard "not computable" case above, matching the original's
                `if sxx > 0 and resid_std > 0 else scar = nan` nuance.
            L : int, event-window length (len(offsets))
    """
    est_s = get_trading_day_offset(trading_cal, ref_date, est_start, anchor=anchor)
    est_e = get_trading_day_offset(trading_cal, ref_date, est_end, anchor=anchor)
    evt_s = get_trading_day_offset(trading_cal, ref_date, a, anchor=anchor)
    evt_e = get_trading_day_offset(trading_cal, ref_date, b, anchor=anchor)

    if any(d is None for d in [est_s, est_e, evt_s, evt_e]):
        return None

    model = estimate_market_model(firm_rets, market_rets, est_s, est_e)
    if model is None:
        return None
    alpha, beta, resid_std, n_est, x_bar, sxx = model

    n_days = b - a + 1
    evt_days = trading_cal[(trading_cal >= evt_s) & (trading_cal <= evt_e)]
    ar_series = (firm_rets.reindex(evt_days) - (alpha + beta * market_rets.reindex(evt_days))).dropna()

    if len(ar_series) != n_days:
        return None

    offsets = list(range(a, b + 1))
    ar = ar_series.to_numpy()

    car = ar_series.sum()
    L = len(ar_series)
    mkt_evt = market_rets.reindex(ar_series.index)
    if sxx > 0 and resid_std > 0:
        forecast_factor = L + (L ** 2) / n_est + ((mkt_evt.sum() - L * x_bar) ** 2) / sxx
        scar = car / (resid_std * np.sqrt(forecast_factor))
    else:
        scar = np.nan

    return dict(
        alpha=alpha, beta=beta, resid_std=resid_std, n_est=n_est,
        offsets=offsets, ar=ar, car=car, scar=scar, L=L,
        event_days=evt_days,
    )


def bmp_test(scar_values):
    """
    Boehmer, Musumeci & Poulsen (1991) standardized cross-sectional test.
    Ported verbatim from 31_event_study.ipynb. One-sided, H1: mean SCAR < 0.

    Returns (t_stat, p_value); (nan, nan) if fewer than 2 valid observations.
    """
    scar = np.asarray(scar_values, dtype=float)
    scar = scar[~np.isnan(scar)]
    n = len(scar)
    if n < 2:
        return np.nan, np.nan
    mean_scar = scar.mean()
    std_scar = scar.std(ddof=1)
    t_stat = np.sqrt(n) * mean_scar / std_scar
    p_value = stats.t.cdf(t_stat, df=n - 1)
    return t_stat, p_value


def cluster_robust_mean_test_fast(values, clusters):
    """
    Closed-form one-sample cluster-robust t-test of H0: mean(values) = 0 vs
    H1: mean < 0, standard errors clustered by `clusters`. Avoids the
    per-call overhead of `statsmodels.formula.api.ols(cov_type='cluster')`
    so it is cheap enough for a Monte Carlo inner loop (planted-effect power
    simulation, firm-clustered bootstrap).

    For an intercept-only OLS, mean = ybar and the cluster-robust variance is
    the standard Liang-Zeger / CR1 "sum of squared cluster score sums"
    sandwich estimator (matching statsmodels' sandwich_covariance.cov_cluster
    with its default small-sample correction, which scales by
    (n_groups/(n_groups-1)) * ((nobs-1)/(nobs-k_params)) with k_params=1 for
    an intercept-only model -- NOT (nobs-n_groups), a common point of
    confusion):
        resid_i = y_i - ybar
        u_g     = sum of resid_i for cluster g
        V       = (G/(G-1)) * ((n-1)/(n-1)) * sum_g(u_g**2) / n**2
        se      = sqrt(V)
        t       = ybar / se
        p       = P(T_{G-1} <= t)   (one-sided, H1: mean < 0)

    Validated to match smf.ols(cov_type='cluster') exactly (diff < 1e-12 on
    mean/se/t/p) in the notebook's engine-validation cell.

    Parameters
    ----------
    values : array-like, shape (n,)
    clusters : array-like, shape (n,), cluster/group id per observation

    Returns
    -------
    dict(mean, se, t_stat, p_value, n_clusters, n_obs) -- nan fields if the
    test is not computable (fewer than 2 clusters, or n <= G).
    """
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    valid = ~np.isnan(values)
    values = values[valid]
    clusters = clusters[valid]
    n = len(values)

    uniq, inv = np.unique(clusters, return_inverse=True)
    G = len(uniq)
    if G < 2 or n <= G:
        return dict(mean=np.nan, se=np.nan, t_stat=np.nan, p_value=np.nan,
                     n_clusters=G, n_obs=n)

    ybar = values.mean()
    resid = values - ybar
    u_g = np.bincount(inv, weights=resid, minlength=G)
    # k_params = 1 (intercept-only model), so (n-1)/(n-k_params) = 1 and the
    # correction reduces to G/(G-1) -- matches statsmodels' cov_cluster exactly.
    k_params = 1
    small_sample = (G / (G - 1)) * ((n - 1) / (n - k_params))
    var = small_sample * np.sum(u_g ** 2) / (n ** 2)
    se = np.sqrt(var)
    if se <= 0 or not np.isfinite(se):
        return dict(mean=ybar, se=se, t_stat=np.nan, p_value=np.nan,
                     n_clusters=G, n_obs=n)
    t_stat = ybar / se
    p_value = stats.t.cdf(t_stat, df=G - 1)
    return dict(mean=ybar, se=se, t_stat=t_stat, p_value=p_value,
                n_clusters=G, n_obs=n)
