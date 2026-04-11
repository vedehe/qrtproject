"""
QRT Quant Challenge - Vol-Neutral Concentrated Strategy

Pipeline:
  1. Stable features -> rank ALL stocks -> EWMA(3) -> cross-sectional normalize -> IC_IR weighted sum
  2. Universe filter + Portfolio-level EWMA(5) for turnover reduction
  3. Volatility-group neutralization (10 groups by vol_20)
  4. True sleeve concentration (keep only top/bottom 12% by signal strength)
  5. Normalize long and short sleeves separately + max weight clipping (0.1)
"""

import pandas as pd
import numpy as np
import datetime
from utils import get_universe_adjusted_series

# Configuration
SIGNAL_CONFIGS = [
    ('trend_5_20',                    -0.1023),
    ('accumulation_distribution_index',-0.0990),
    ('trend_20_60',                   -0.0944),
    ('average_true_range',            -0.0830),
    ('trend_1_3',                     -0.0803),
    ('stochastic_oscillator',         -0.0734),
    ('on_balance_volume',              0.0597),
    ('ultimate_oscillator',           -0.0572),
    ('relative_strength_index',       -0.0532),
    ('volatility_20',                 -0.0511),
    ('ease_of_movement',              -0.0499),
    ('volume',                         0.0487),
    ('volatility_60',                 -0.0484),
    ('chaikin_money_flow',            -0.0453),
    ('chande_momentum_oscillator',    -0.0389),
    ('ichimoku',                      -0.0339),
    ('commodity_channel_index',       -0.0279),
    ('aroon',                         -0.0189),
]

FEAT_EWMA = 3
PORTFOLIO_EWMA = 5
VOL_GROUPS = 10
CONCENTRATION_PCT = 0.12   # keep top/bottom 12%
MAX_WEIGHT = 0.1
VOL_FEATURE = 'volatility_20'


def clip_and_balance_series(weights: pd.Series, max_weight: float = MAX_WEIGHT) -> pd.Series:
    pos = weights[weights > 0].copy()
    neg = (-weights[weights < 0]).copy()

    if len(pos) > 0:
        for _ in range(10):
            if pos.max() <= max_weight + 1e-12:
                break
            pos = pos.clip(upper=max_weight)
            pos_sum = pos.sum()
            if pos_sum > 0:
                pos *= 0.5 / pos_sum
        pos_sum = pos.sum()
        if pos_sum > 0:
            pos *= 0.5 / pos_sum

    if len(neg) > 0:
        for _ in range(10):
            if neg.max() <= max_weight + 1e-12:
                break
            neg = neg.clip(upper=max_weight)
            neg_sum = neg.sum()
            if neg_sum > 0:
                neg *= 0.5 / neg_sum
        neg_sum = neg.sum()
        if neg_sum > 0:
            neg *= 0.5 / neg_sum

    balanced = pd.Series(0.0, index=weights.index)
    if len(pos) > 0:
        balanced.loc[pos.index] = pos
    if len(neg) > 0:
        balanced.loc[neg.index] = -neg
    return balanced


# Vectorized Portfolio Generation
def generate_portfolio_vectorized(
    entire_features: pd.DataFrame,
    universe: pd.DataFrame,
    start_date: str,
    end_date: str,
):
    try:
        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        cutoff_date = datetime.datetime.strptime('2005-01-01', '%Y-%m-%d')
    except ValueError:
        raise ValueError("start_date and end_date must be strings in 'YYYY-MM-DD' format.")
    if start_dt >= end_dt:
        raise ValueError("start_date must be earlier than end_date.")
    if start_dt < cutoff_date:
        raise ValueError("start_date must be later than '2005-01-01'.")

    trading_days = universe.index[(universe.index >= start_dt) & (universe.index <= end_dt)]
    if len(trading_days) == 0:
        raise ValueError("No Trading Days in the specified dates")

    ub = universe.loc[:end_date].astype(bool)

    # Step 1: Raw signal — rank ALL stocks (no universe filter for matching)
    combined = pd.DataFrame(0.0, index=universe.loc[:end_date].index, columns=universe.columns)
    for feat_name, weight in SIGNAL_CONFIGS:
        feat_shifted = entire_features[feat_name].shift(1).loc[:end_date]
        feat_ranked = feat_shifted.rank(axis=1, method='average')
        feat_smoothed = feat_ranked.ewm(span=FEAT_EWMA, min_periods=1).mean()
        signal = feat_smoothed.sub(feat_smoothed.mean(axis=1), axis=0)
        signal = signal.div(signal.abs().sum(axis=1), axis=0).fillna(0)
        combined += weight * signal

    # Step 2: Portfolio-level EWMA (no universe filter — ensures iterative match)
    combined = combined.ewm(span=PORTFOLIO_EWMA, min_periods=1).mean()

    # Step 3: Volatility-group neutralization (universe filter applied here)
    combined = combined.where(ub, np.nan)
    vol_feat = entire_features[VOL_FEATURE].shift(1).loc[:end_date]
    vol_ranked = vol_feat.where(ub, np.nan).rank(axis=1, pct=True)

    for g in range(VOL_GROUPS):
        lo, hi = g / VOL_GROUPS, (g + 1) / VOL_GROUPS
        mask = (vol_ranked >= lo) & (vol_ranked < hi)
        group_sig = combined.where(mask, np.nan)
        group_mean = group_sig.mean(axis=1)
        combined = combined - mask.astype(float) * group_mean.values[:, None]

    # Step 4: True concentration (keep only top/bottom stocks)
    sig_for_rank = combined.where(ub, np.nan)
    pct_rank = sig_for_rank.rank(axis=1, pct=True)
    half = CONCENTRATION_PCT / 2
    long_mask = pct_rank >= 1 - half
    short_mask = pct_rank <= half

    # Step 5: Normalize long and short sleeves separately to preserve sparsity
    long_scores = combined.where(long_mask, 0).clip(lower=0)
    short_scores = (-combined.where(short_mask, 0)).clip(lower=0)

    long_sum = long_scores.sum(axis=1).replace(0, np.nan)
    short_sum = short_scores.sum(axis=1).replace(0, np.nan)
    portfolio = long_scores.div(long_sum, axis=0).fillna(0) * 0.5
    portfolio = portfolio - short_scores.div(short_sum, axis=0).fillna(0) * 0.5

    if (portfolio.abs().max(axis=1) > MAX_WEIGHT + 1e-8).any():
        portfolio = portfolio.apply(clip_and_balance_series, axis=1)

    return portfolio.fillna(0).loc[start_date:end_date]


# Iterative Per-Day Weight Computation
def get_weights(features: pd.DataFrame, today_universe: pd.Series) -> dict:
    if features.shape[0] == 0:
        return (today_universe * 1).replace(0, np.nan).dropna().fillna(0).to_dict()

    lookback = min(features.shape[0], 40)
    recent = features.iloc[-lookback:]

    # Step 1: Raw signal — rank ALL stocks (no universe filter, matching vectorized)
    combined_hist = pd.DataFrame(0.0, index=recent.index, columns=today_universe.index)
    for feat_name, weight in SIGNAL_CONFIGS:
        feat_data = recent.xs(feat_name, axis=1, level=0)
        feat_ranked = feat_data.rank(axis=1, method='average')
        feat_smoothed = feat_ranked.ewm(span=FEAT_EWMA, min_periods=1).mean()
        signal = feat_smoothed.sub(feat_smoothed.mean(axis=1), axis=0)
        signal = signal.div(signal.abs().sum(axis=1), axis=0).fillna(0)
        combined_hist += weight * signal

    # Step 2: Portfolio EWMA (no universe filter — matches vectorized)
    combined_smooth = combined_hist.ewm(span=PORTFOLIO_EWMA, min_periods=1).mean()
    combined = combined_smooth.iloc[-1]

    # Step 3: Volatility-group neutralization (universe filter applied here)
    combined = get_universe_adjusted_series(combined, today_universe)
    vol_data = recent.xs(VOL_FEATURE, axis=1, level=0).iloc[-1]
    vol_adj = get_universe_adjusted_series(vol_data, today_universe)
    vol_rank_pct = vol_adj.rank(pct=True)

    for g in range(VOL_GROUPS):
        lo, hi = g / VOL_GROUPS, (g + 1) / VOL_GROUPS
        mask = (vol_rank_pct >= lo) & (vol_rank_pct < hi)
        group_vals = combined.where(mask, np.nan)
        group_mean = group_vals.mean()
        if not np.isnan(group_mean):
            combined = combined - mask.astype(float) * group_mean

    # Step 4: True concentration (rank only universe stocks via NaN exclusion)
    sig_rank_pct = combined.rank(pct=True)
    half = CONCENTRATION_PCT / 2
    long_mask = sig_rank_pct >= 1 - half
    short_mask = sig_rank_pct <= half

    # Step 5: Normalize long and short sleeves separately to preserve sparsity
    long_scores = combined.where(long_mask, 0).clip(lower=0)
    short_scores = (-combined.where(short_mask, 0)).clip(lower=0)

    alpha = pd.Series(0.0, index=combined.index)
    long_sum = long_scores.sum()
    short_sum = short_scores.sum()
    if long_sum > 0:
        alpha.loc[long_scores[long_scores > 0].index] = long_scores[long_scores > 0] / long_sum * 0.5
    if short_sum > 0:
        alpha.loc[short_scores[short_scores > 0].index] = -short_scores[short_scores > 0] / short_sum * 0.5

    if alpha.abs().max() > MAX_WEIGHT + 1e-8:
        alpha = clip_and_balance_series(alpha)

    alpha = alpha.replace(0, np.nan).dropna()
    return alpha.to_dict()
