"""Generate submission, backtest, and verify match."""
import pandas as pd
import numpy as np
import datetime
import os, sys
sys.path.insert(0, '/Users/adityaanand/dev/qrt')

from utils import (get_universe_adjusted_series, scale_to_book_long_short,
                    generate_portfolio, backtest_portfolio, match_implementations)
from strategy import (generate_portfolio_vectorized, get_weights,
                       SIGNAL_CONFIGS, FEAT_EWMA, PORTFOLIO_EWMA,
                       VOL_GROUPS, CONCENTRATION_PCT)

DATA_DIR = '/Users/adityaanand/dev/data/'
features = pd.read_parquet(DATA_DIR + 'features.parquet')
universe = pd.read_parquet(DATA_DIR + 'universe.parquet')
returns = pd.read_parquet(DATA_DIR + 'returns.parquet')

print(f"Strategy: {len(SIGNAL_CONFIGS)} features, feat_EWMA={FEAT_EWMA}, port_EWMA={PORTFOLIO_EWMA}")
print(f"  Vol groups={VOL_GROUPS}, concentration={CONCENTRATION_PCT:.0%}")

# 1. Generate full portfolio
print("\n=== Generating vectorized portfolio (2005-2025) ===")
portfolio = generate_portfolio_vectorized(features, universe, "2005-01-03", "2025-02-07")
print(f"Portfolio shape: {portfolio.shape}")

# 2. Backtest on training periods
print("\n=== Backtest: Full 2005-2019 ===")
sr_all, pnl_all = backtest_portfolio(portfolio.loc[:"2019"], returns.loc[:"2019"], universe.loc[:"2019"], False, True)

print("\n=== Backtest: 2005-2009 ===")
sr_05, _ = backtest_portfolio(portfolio.loc["2005":"2009"], returns.loc["2005":"2009"], universe.loc["2005":"2009"], False, True)

print("\n=== Backtest: 2010-2014 ===")
sr_10, _ = backtest_portfolio(portfolio.loc["2010":"2014"], returns.loc["2010":"2014"], universe.loc["2010":"2014"], False, True)

print("\n=== Backtest: 2015-2019 ===")
sr_15, _ = backtest_portfolio(portfolio.loc["2015":"2019"], returns.loc["2015":"2019"], universe.loc["2015":"2019"], False, True)

print(f"\n=== Summary ===")
print(f"Overall:   Net Sharpe = {sr_all}")
print(f"2005-2009: Net Sharpe = {sr_05}")
print(f"2010-2014: Net Sharpe = {sr_10}")
print(f"2015-2019: Net Sharpe = {sr_15}")

# 3. Verify match
print("\n=== Verifying iterative/vectorized match ===")
match_implementations(get_weights, portfolio, features, universe, returns)

# 4. Save submission
portfolio.to_csv("/Users/adityaanand/dev/qrt/submission.csv")
print(f"\nSubmission saved: {portfolio.shape}")
