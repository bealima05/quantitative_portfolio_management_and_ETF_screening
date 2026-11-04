import re
from functools import reduce
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm


class ETFScreener:
    def __init__(self, file2, ticker_col="etf", track_col="track",
                 ter_col="ter", class_col="class", rf_col="risk free",
                 log_returns=True):
        self.file2       = file2
        self.ticker_col  = ticker_col
        self.track_col   = track_col
        self.ter_col     = ter_col
        self.class_col   = class_col
        self.rf_col      = rf_col
        self.log_returns = log_returns

        # ---- populated by read() ----
        self.track_dict   = None
        self.ter_dict     = None
        self.df_ter       = None
        self.df_rf        = None
        self.class_dict   = None
        self.weights_dict = None

        # ---- populated by extract() ----
        self.close_etf_df   = None
        self.volume_etf_df  = None
        self.close_track_df = None
        self.difference     = None

        # ---- metric frames ----
        self.df_stats   = None
        self.df_beta    = None
        self.df_volume  = None
        self.bid        = None
        self.df_spread  = None
        self.df_te      = None
        self.df_class   = None
        self.df_metrics = None

        # ---- scoring ----
        self.df_scoring  = None
        self.df_eligible = None
        self.df_score    = None

        # ---- allocation ----
        self.df_class_allocation = None
        self.top_picks           = None

    # ------------------------------------------------------------------ #
    # 1. Load input file
    # ------------------------------------------------------------------ #
    def read(self):
        df = pd.read_excel(self.file2)

        df[self.ticker_col] = df[self.ticker_col].astype(str).str.strip()
        df[self.track_col]  = df[self.track_col].astype(str).str.strip()
        df[self.class_col]  = df[self.class_col].astype(str).str.strip()
        df[self.ter_col]    = pd.to_numeric(df[self.ter_col], errors="coerce")
        df[self.rf_col]     = pd.to_numeric(df[self.rf_col], errors="coerce")

        df = df.sort_values(by=self.ticker_col)

        self.track_dict = dict(zip(df[self.ticker_col], df[self.track_col]))
        self.ter_dict   = dict(zip(df[self.ticker_col], df[self.ter_col]))
        self.class_dict = dict(zip(df[self.ticker_col], df[self.class_col]))

        self.df_ter = df[[self.ticker_col, self.track_col, self.ter_col]].copy()
        self.df_rf  = df[[self.ticker_col, self.track_col, self.rf_col]].copy()

        # weights per profile (columns named "DC<n>")
        self.weights_dict = {
            f"DC{m.group(1)}": df[col].to_numpy()
            for col in df.columns
            for m in [re.search(r"DC\s*(\d+)", col, re.IGNORECASE)]
            if m
        }
        return self

    # ------------------------------------------------------------------ #
    # 2. Download prices
    # ------------------------------------------------------------------ #
    def extract(self, start_date, end_date):
        self.difference = (datetime.strptime(end_date, "%Y-%m-%d")
                            - datetime.strptime(start_date, "%Y-%m-%d"))

        etfs   = list(self.track_dict.keys())
        tracks = list(dict.fromkeys(self.track_dict.values()))
        all_tickers = sorted(dict.fromkeys(etfs + tracks))

        data = yf.download(all_tickers, start=start_date, end=end_date,
                           progress=False, auto_adjust=True)

        close  = data["Close"]
        volume = data["Volume"]
        if isinstance(close, pd.Series):
            close = close.to_frame()
        if isinstance(volume, pd.Series):
            volume = volume.to_frame()

        close  = close.reindex(columns=all_tickers)
        volume = volume.reindex(columns=all_tickers)

        self.close_etf_df   = close.reindex(columns=etfs)
        self.close_track_df = close.reindex(columns=tracks)
        self.volume_etf_df  = volume.reindex(columns=etfs)
        return self

    # ------------------------------------------------------------------ #
    # Helper: aligned return series for a single (etf, benchmark) pair
    # ------------------------------------------------------------------ #
    def _paired_returns(self, etf, bm):
        data = pd.concat([self.close_etf_df[etf], self.close_track_df[bm]], axis=1)
        data.columns = [etf, bm]
        data = data.dropna()

        if self.log_returns:
            returns = np.log(data / data.shift(1))
        else:
            returns = data.pct_change()

        return returns.dropna()

    # ------------------------------------------------------------------ #
    # 3a. Statistics (mean, vol, correlation per pair)
    # ------------------------------------------------------------------ #
    def statistics(self):
        rows = []
        for etf, bm in self.track_dict.items():
            try:
                returns = self._paired_returns(etf, bm)
                if len(returns) < 2:
                    raise ValueError("Insufficient returns")
                etf_ret, bm_ret = returns[etf], returns[bm]
                rows.append({
                    "etf": etf, "track": bm,
                    "mean": etf_ret.mean(),
                    "volatility": etf_ret.std(),
                    "correlation": etf_ret.corr(bm_ret),
                    "n_obs": len(returns),
                })
            except Exception:
                rows.append({"etf": etf, "track": bm, "correlation": np.nan, "n_obs": np.nan})

        self.df_stats = pd.DataFrame(rows)
        return self.df_stats

    # ------------------------------------------------------------------ #
    # 3b. Beta / R^2 per pair (cov/var beta + OLS R^2)
    # ------------------------------------------------------------------ #
    def beta(self):
        rows = []
        for etf, bm in self.track_dict.items():
            try:
                returns = self._paired_returns(etf, bm)
                if len(returns) < 2:
                    raise ValueError("Insufficient returns")

                y, x = returns[etf], returns[bm]
                var_x = x.var(ddof=1)
                if pd.isna(var_x) or var_x == 0:
                    raise ValueError("Benchmark variance is zero/invalid")

                beta_cov = y.cov(x) / var_x
                model = sm.OLS(y, sm.add_constant(x)).fit()

                rows.append({
                    "etf": etf, "track": bm,
                    "beta": beta_cov,
                    "r_squared": model.rsquared,
                })
            except Exception:
                rows.append({"etf": etf, "track": bm, "beta": np.nan, "r_squared": np.nan})

        self.df_beta = pd.DataFrame(rows)
        return self.df_beta

    # ------------------------------------------------------------------ #
    # 3c. Volume metrics
    # ------------------------------------------------------------------ #
    def volume_metrics(self):
        rows = []
        for etf, bm in self.track_dict.items():
            try:
                data = pd.concat([self.close_etf_df[etf], self.volume_etf_df[etf]], axis=1).dropna()
                data.columns = ["close", "volume"]
                dtvv_series = data["close"] * data["volume"]

                rows.append({
                    "etf": etf, "track": bm,
                    "volume_mean": data["volume"].rolling(self.difference).mean().iloc[-1],
                    "dtvv": dtvv_series.iloc[-1],
                    "dtvv_mean": dtvv_series.rolling(self.difference).mean().iloc[-1],
                })
            except Exception:
                rows.append({"etf": etf, "track": bm, "dtvv_mean": np.nan})

        self.df_volume = pd.DataFrame(rows)
        return self.df_volume

    # ------------------------------------------------------------------ #
    # 3d. Bid-offer spread (live quotes)
    # ------------------------------------------------------------------ #
    def bid_offer_spread(self):
        rows = []
        bid = None
        for etf, bm in self.track_dict.items():
            try:
                info = yf.Ticker(etf).info
                bid, ask = info.get("bid"), info.get("ask")
                if bid is None or ask is None or bid <= 0 or ask <= 0:
                    raise ValueError("Invalid bid/ask")
                spread_pct = (ask - bid) / ask * 100
                rows.append({"etf": etf, "track": bm, "spread_pct": spread_pct})
            except Exception:
                rows.append({"etf": etf, "track": bm, "spread_pct": None})

        self.bid = bid
        self.df_spread = pd.DataFrame(rows)
        return self.bid, self.df_spread

    # ------------------------------------------------------------------ #
    # 3e. Tracking error
    # ------------------------------------------------------------------ #
    def tracking_error(self):
        rows = []
        for etf, bm in self.track_dict.items():
            etf_close, bm_close = self.close_etf_df[etf], self.close_track_df[bm]
            if self.log_returns:
                etf_ret = np.log(etf_close / etf_close.shift(1))
                bm_ret  = np.log(bm_close / bm_close.shift(1))
            else:
                etf_ret = etf_close.pct_change()
                bm_ret  = bm_close.pct_change()
            rows.append({"etf": etf, "track": bm, "te": (etf_ret - bm_ret).std()})

        self.df_te = pd.DataFrame(rows)
        return self.df_te

    # ------------------------------------------------------------------ #
    # 3f. Compute + merge all metrics into a single table
    # ------------------------------------------------------------------ #
    def compute_metrics(self):
        self.statistics()
        self.beta()
        self.volume_metrics()
        self.bid_offer_spread()
        self.tracking_error()

        self.df_class = pd.DataFrame({
            "etf":   list(self.track_dict.keys()),
            "track": list(self.track_dict.values()),
            "class": [self.class_dict[etf] for etf in self.track_dict.keys()],
        })

        dfs = [self.df_class, self.df_stats, self.df_beta,
               self.df_ter, self.df_te, self.df_spread, self.df_volume]
        self.df_metrics = reduce(
            lambda left, right: pd.merge(left, right, on=["etf", "track"], how="outer"), dfs
        )
        return self.df_metrics

    def save_metrics(self, path="excels_2/etf_metrics_full.xlsx"):
        self.df_metrics.to_excel(path, index=False)
        print(f"Saved full metrics table to {path}")

    # ------------------------------------------------------------------ #
    # 4. Eligibility filters (vectorized across the whole frame)
    # ------------------------------------------------------------------ #
    def apply_eligibility(self, corr_min=0.95, beta_min=0.8, beta_max=1.2,
                          ter_max=0.005, te_max=None, spread_max=1.5,
                          dtvv_min=40000, group_col="scoring_group"):
        if te_max is None:
            te_max = {"equity": 0.01, "bonds": 0.01}

        # --- scoring_groups step ---
        df = self.df_metrics.copy()
        df[self.class_col] = df[self.class_col].astype("string").str.strip().str.lower()
        df[self.class_col] = df[self.class_col].replace(
            {"": pd.NA, "nan": pd.NA, "none": pd.NA, "null": pd.NA})
        df[group_col] = df[self.class_col]
        self.df_scoring = df

        # --- eligibility step, vectorized via group-mapped thresholds ---
        def threshold_series(threshold):
            """Broadcast a scalar or {group: value} threshold to a per-row Series."""
            if isinstance(threshold, dict):
                return df[group_col].map(threshold)
            return pd.Series(threshold, index=df.index)

        fail_corr   = df["correlation"] < threshold_series(corr_min)
        fail_beta   = (df["beta"] < threshold_series(beta_min)) | (df["beta"] > threshold_series(beta_max))
        fail_ter    = df["ter"] > threshold_series(ter_max)
        fail_te     = df["te"] > threshold_series(te_max)
        fail_spread = df["spread_pct"] > threshold_series(spread_max)
        fail_dtvv   = df["dtvv_mean"] < threshold_series(dtvv_min)

        fail_df = pd.DataFrame({
            "corr": fail_corr.fillna(False), "beta": fail_beta.fillna(False),
            "ter": fail_ter.fillna(False), "te": fail_te.fillna(False),
            "spread": fail_spread.fillna(False), "dtvv": fail_dtvv.fillna(False),
        })

        df["eligible"] = ~fail_df.any(axis=1)
        df["non_eligibility_reason"] = fail_df.apply(
            lambda row: "doesn't apply" if not row.any() else ";".join(row.index[row]), axis=1
        )

        self.df_eligible = df
        return self.df_eligible

    # ------------------------------------------------------------------ #
    # Scoring transforms (pure, used by score())
    # ------------------------------------------------------------------ #
    @staticmethod
    def score_higher_better(values, decimals, floor, cap):
        if cap <= floor:
            raise ValueError("'cap' must be greater than 'floor'.")
        x = values if isinstance(values, pd.Series) else pd.Series(values)
        return ((x - floor) / (cap - floor)).clip(0, 1).round(decimals)

    @staticmethod
    def score_lower_better(values, decimals, max, cap):
        if cap <= max:
            raise ValueError("'cap' must be greater than 'max'.")
        x = values if isinstance(values, pd.Series) else pd.Series(values)
        return ((cap - x) / (cap - max)).clip(0, 1).round(decimals)

    @staticmethod
    def score_target_better(values, decimals, target, tolerance, max_deviation):
        if tolerance < 0:
            raise ValueError("'tolerance' must be non-negative.")
        if max_deviation <= tolerance:
            raise ValueError("'max_deviation' must be greater than 'tolerance'.")
        x = values if isinstance(values, pd.Series) else pd.Series(values)
        distance = (x - target).abs()
        score = (max_deviation - distance) / (max_deviation - tolerance)
        score = score.where(distance > tolerance, 1.0)
        return score.clip(0, 1).round(decimals)

    # ------------------------------------------------------------------ #
    # 5. Combined scoring
    # ------------------------------------------------------------------ #
    def score(self, decimals=3,
              repl_weights=(0.3, 0.4, 0.3),     # corr, beta, te
              liq_weights=(0.6, 0.4),           # dtvv, spread
              pillar_weights=(0.34, 0.33, 0.33) # replication, cost, liquidity
              ):
        df = self.df_eligible

        df["score_corr"]   = self.score_higher_better(df["correlation"], decimals, floor=0.95, cap=0.99)
        df["score_volume"] = self.score_higher_better(df["dtvv_mean"], decimals, floor=50000, cap=1000000)
        df["score_ter"]    = self.score_lower_better(df["ter"], decimals, max=0.001, cap=0.005)
        df["score_te"]     = self.score_lower_better(df["te"], decimals, max=0.001, cap=0.01)
        df["score_spread"] = self.score_lower_better(df["spread_pct"], decimals, max=0.1, cap=1.5)
        df["score_beta"]   = self.score_target_better(df["beta"], decimals, target=1.0,
                                                       tolerance=0.01, max_deviation=0.20)

        corr_w, beta_w, te_w = repl_weights
        df["score_replication"] = corr_w * df["score_corr"] + beta_w * df["score_beta"] + te_w * df["score_te"]
        df["score_cost"] = 1.0 * df["score_ter"]

        dtvv_w, spread_w = liq_weights
        df["score_liquidity"] = dtvv_w * df["score_volume"] + spread_w * df["score_spread"]

        repl_w, cost_w, liq_w = pillar_weights
        df["total_score"] = (repl_w * df["score_replication"]
                             + cost_w * df["score_cost"]
                             + liq_w * df["score_liquidity"])

        df["rank_in_group"] = df.groupby("scoring_group")["total_score"].rank(method="dense", ascending=False)
        df = df.sort_values(by=["scoring_group", "rank_in_group"])
        self.df_eligible = df

        self.df_score = pd.DataFrame({
            "etf":               df["etf"],
            "track":             df["track"],
            "class":             df["scoring_group"],
            "score replication": df["score_replication"],
            "score cost":        df["score_cost"],
            "score liquidity":   df["score_liquidity"],
            "total_score":       df["total_score"],
            "eligible":          df["eligible"],
            "reason":            df["non_eligibility_reason"],
            "rank":              df["rank_in_group"],
        })
        return self.df_score

    def save_scores(self, path="excels_2/score3.xlsx"):
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for group, data in self.df_score.groupby("class"):
                data.to_excel(writer, sheet_name=str(group), index=False)
        print(f"Saved scoring table to {path}")

    # ------------------------------------------------------------------ #
    # 6. Class allocation per profile (Table B)
    # ------------------------------------------------------------------ #
    def class_allocation(self):
        self.top_picks = self.df_eligible.loc[
            self.df_eligible["rank_in_group"] == 1, ["etf", "scoring_group"]]

        all_etfs_ordered = list(self.track_dict.keys())

        alloc_rows = []
        for profile, weights in self.weights_dict.items():
            class_alloc = {}
            for etf, w in zip(all_etfs_ordered, weights):
                if w == 0:
                    continue
                cls = self.class_dict.get(etf, "unknown")
                class_alloc[cls] = class_alloc.get(cls, 0.0) + float(w)
            for cls, w in class_alloc.items():
                alloc_rows.append({"profile": profile, "class": cls, "target_weight": w})

        self.df_class_allocation = pd.DataFrame(alloc_rows)
        return self.df_class_allocation

    def save_class_allocation(self, path="excels_2/class_allocation.xlsx"):
        self.df_class_allocation.to_excel(path, index=False)
        print(f"Saved class allocation table to {path}")
