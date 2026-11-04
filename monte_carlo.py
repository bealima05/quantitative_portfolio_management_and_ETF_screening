import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MonteCarlo:
    """
    Monte Carlo (GBM) forecasting of a buy and hold portfolio built from several
    correlated assets, with Expected Maximum Drawdown analysis.

    Pipeline:
      1. simulate()          correlated GBM asset paths via Cholesky of the covariance.
      2. portfolio paths     static target weights fix the units at t=0 (buy and hold);
                             NAV[t] = V0 * sum_i w_i * G_i[t], with G_i the asset gross growth.
      3. analyse()           maximum drawdown distribution of the portfolio and per asset.
      4. convergence(), sensitivity(), plots.

    Parameters of note:
      mean : per asset daily mean log return (Series indexed by asset).
      cov  : per asset daily covariance matrix (DataFrame, assets x assets).
             Per asset sigma is taken as sqrt(diag(cov)) so the marginal vols and the
             correlations come from the same matrix.
      var  : kept for signature compatibility, not used (sigma derived from cov).
    Weights are static (df_profile[profile]), fixed at t=0, consistent with buy and hold.
    """

    def __init__(self, df_profile, profile, df_returns, mean, var, cov, V0, **kwargs):
        self.df_profile = df_profile
        self.profile = profile
        self.df_ret = df_returns
        self.mu = mean
        self.var = var
        self.cov = cov
        self.V0 = V0

        self.seed = kwargs.get('seed')
        self.th = kwargs.get('time horizon', 252)
        self.paths = kwargs.get('number of paths', 5000)
        self.cutoff_date = kwargs.get('cutoff_date')  # pd.Timestamp, day 0 of the simulation

        # Aligned inputs (filled by get_weights / _prepare_params)
        self.assets = None
        self.weights = None
        self.mu_vec = None
        self.cov_mat = None
        self.sigma_vec = None

        # Simulation outputs
        self.asset_growth = None   # (th+1, paths, n_assets), starts at 1.0
        self.ptf_nav = None        # (th+1, paths)

        # Ex-ante outputs
        self.ew = None             # (th+1, paths, n_assets) exposure weights per day/path
        self.ea_ptf_ret = None     # (th+1, paths) ex-ante portfolio return per day/path

        # Analysis outputs
        self.ptf_mdd = None        # per path portfolio max drawdown (<= 0)
        self.summary = None        # DataFrame: portfolio + per asset metrics


    # ------------------------------------------------------------------ setup
    def get_weights(self):
        """Static target weights from the profile; defines the active asset order."""
        w = self.df_profile[self.profile]
        w = w[w > 0]
        self.assets = w.index.tolist()
        self.weights = w.reindex(self.assets).to_numpy(dtype=float)


    def _prepare_params(self):
        """Align mu and cov to the active asset order; derive per asset sigma."""
        if self.assets is None:
            self.get_weights()
        self.mu_vec = self.mu.reindex(self.assets).to_numpy(dtype=float)
        self.cov_mat = self.cov.reindex(index=self.assets, columns=self.assets).to_numpy(dtype=float)
        self.sigma_vec = np.sqrt(np.diag(self.cov_mat))


    # -------------------------------------------------------------- simulation
    def simulate(self, mu_vec=None, cov_mat=None, n_paths=None, rng=None):
        """
        Correlated GBM asset paths and the resulting buy and hold portfolio NAV.
        Optional overrides (mu_vec, cov_mat, n_paths) are used by sensitivity().
        """
        if self.weights is None:
            self._prepare_params()

        mu_vec = self.mu_vec if mu_vec is None else mu_vec
        cov_mat = self.cov_mat if cov_mat is None else cov_mat
        n_paths = self.paths if n_paths is None else n_paths
        rng = np.random.default_rng(self.seed) if rng is None else rng

        n_assets = len(self.assets)
        T = self.th

        L = np.linalg.cholesky(cov_mat)                       # cov = L @ L.T
        Z = rng.normal(0.0, 1.0, (T, n_paths, n_assets))
        shocks = Z @ L.T                                      # Cov(shocks) = cov_mat

        drift = mu_vec - 0.5 * np.diag(cov_mat)               # daily, per asset
        log_incr = drift + shocks
        log_paths = np.cumsum(log_incr, axis=0)
        growth = np.exp(log_paths)                            # (T, paths, assets)
        growth = np.concatenate([np.ones((1, n_paths, n_assets)), growth], axis=0)

        nav = self.V0 * (growth @ self.weights)               # (T+1, paths)

        # Store only the base simulation (so sensitivity overrides do not overwrite it)
        if mu_vec is self.mu_vec and cov_mat is self.cov_mat and n_paths == self.paths:
            self.asset_growth = growth
            self.ptf_nav = nav
        return growth, nav


    # ----------------------------------------------------------- ex-ante
    def compute_exposure_weights(self):
        """
        Realized exposure weights per day per path under buy and hold.
        value_i[t] = V0 * w_i * G_i[t]; weight = value_i / NAV.
        These drift over time as the simulated prices move (same as the real
        Weight_ columns), even though the starting allocation is static.
        """
        if self.asset_growth is None:
            self.simulate()
        value = self.asset_growth * self.weights            # (T+1, paths, assets)
        nav = value.sum(axis=2, keepdims=True)
        self.ew = value / nav
        return self.ew


    def ex_ante_returns(self):
        """
        Ex-ante portfolio return per day per path:
            ea[t] = sum_i ew[t-1, i] * mu_i
        Start-of-day (t-1) exposure weights times the per asset expected return.
        In the simulation the expected return is the fixed drift mu_vec used to
        generate the paths, so the only time variation comes from weight drift.
        Day 0 is NaN (no prior allocation).
        """
        if self.ew is None:
            self.compute_exposure_weights()
        ea = np.full((self.th + 1, self.paths), np.nan)
        ea[1:] = (self.ew[:-1] * self.mu_vec).sum(axis=2)   # shift(1) on weights
        self.ea_ptf_ret = ea
        return ea


    def mean_exposure_weights(self):
        """Exposure weights averaged across paths, as a (day x asset) DataFrame."""
        if self.ew is None:
            self.compute_exposure_weights()
        return pd.DataFrame(self.ew.mean(axis=1), columns=self.assets)


    def mean_ex_ante_return(self):
        """Ex-ante portfolio return averaged across paths, as a daily Series."""
        if self.ea_ptf_ret is None:
            self.ex_ante_returns()
        m = np.full(self.th + 1, np.nan)
        m[1:] = self.ea_ptf_ret[1:].mean(axis=1)   # rows 1: are fully populated
        return pd.Series(m, name="ea_ptf_ret")


    # --------------------------------------------------------------- drawdown
    @staticmethod
    def _drawdown_stats(paths):
        """
        Per path fractional max drawdown (<= 0) and duration info.
        paths shape: (steps, n_paths).
        """
        run_max = np.maximum.accumulate(paths, axis=0)
        dd = paths / run_max - 1.0
        mdd = dd.min(axis=0)
        trough = dd.argmin(axis=0)

        n = paths.shape[1]
        peak_to_trough = np.empty(n)
        recovery = np.full(n, np.nan)
        for i in range(n):
            tr = trough[i]
            pk = paths[:tr + 1, i].argmax()
            peak_to_trough[i] = tr - pk
            after = np.where(paths[tr:, i] >= paths[pk, i])[0]
            if after.size:
                recovery[i] = after[0]
        return {"mdd": mdd, "peak_to_trough": peak_to_trough, "recovery": recovery}


    def analytical_emd_zero_drift(self, sigma):
        """Magdon-Ismail & Atiya (2004) mu=0 reference, as a positive fraction."""
        emd_log = np.sqrt(np.pi / 2.0) * sigma * np.sqrt(self.th)
        return 1.0 - np.exp(-emd_log)


    def analyse(self):
        """Builds the portfolio and per asset max drawdown summary table."""
        if self.ptf_nav is None:
            self.simulate()

        rows = []

        # Portfolio
        ps = self._drawdown_stats(self.ptf_nav)
        self.ptf_mdd = ps["mdd"]
        sigma_ptf = float(np.sqrt(self.weights @ self.cov_mat @ self.weights))
        rows.append(self._row("PORTFOLIO", ps, sigma_ptf))

        # Per asset
        for j, name in enumerate(self.assets):
            asset_nav = self.V0 * self.asset_growth[:, :, j]
            s = self._drawdown_stats(asset_nav)
            rows.append(self._row(name, s, self.sigma_vec[j]))

        self.summary = pd.DataFrame(rows).set_index("name")
        return self.summary


    def _row(self, name, stats, sigma):
        mdd = np.abs(stats["mdd"])
        rec = stats["recovery"]
        self.med_mdd = np.median(mdd)
        self.p95_mdd = np.percentile(mdd,95)
        return {
            "name": name,
            "mean_MDD": mdd.mean(),
            "median_MDD": self.med_mdd,
            "p95_MDD": self.p95_mdd,
            "analytical_EMD_zero_drift": self.analytical_emd_zero_drift(sigma),
            "mean_peak_to_trough_days": stats["peak_to_trough"].mean(),
            "pct_not_recovered": np.isnan(rec).mean() * 100.0,
            "mean_recovery_days": np.nanmean(rec) if np.isfinite(rec).any() else np.nan,
        }


    # ----------------------------------------------------------- diagnostics
    def convergence(self, step=50):
        """Running mean of portfolio |MDD| as paths accumulate."""
        if self.ptf_mdd is None:
            self.analyse()
        abs_mdd = np.abs(self.ptf_mdd)
        ns = np.arange(step, len(abs_mdd) + 1, step)
        running = np.array([abs_mdd[:k].mean() for k in ns])
        return ns, running


    def sensitivity(self, sigma_scales=(0.6, 0.8, 1.0, 1.2, 1.4),
                    mu_scales=(0.0, 1.0, 2.0), n_paths=2000):
        """
        Mean portfolio |MDD| over a grid of mu and sigma scalings.
        sigma scales the whole covariance (cov -> s^2 * cov, i.e. sigma -> s * sigma).
        Returns (sigma_scales, mu_scales, grid[mu, sigma]).
        """
        if self.weights is None:
            self._prepare_params()
        rng = np.random.default_rng(self.seed)
        grid = np.empty((len(mu_scales), len(sigma_scales)))
        for i, ms in enumerate(mu_scales):
            for j, ss in enumerate(sigma_scales):
                _, nav = self.simulate(mu_vec=self.mu_vec * ms,
                                       cov_mat=(ss ** 2) * self.cov_mat,
                                       n_paths=n_paths, rng=rng)
                grid[i, j] = np.abs(self._drawdown_stats(nav)["mdd"]).mean()
        return np.array(sigma_scales), list(mu_scales), grid


    # ---------------------------------------------------------------- plots
    def plot_fan(self, realized_nav=None, realized_dates=None, fname="mc_fan.png"):
        """
        Simulated NAV distribution over time (percentile fan) with an optional
        realized NAV path overlaid for backtest validation.

        X-axis uses real calendar dates if self.cutoff_date is set: simulated
        days are business days starting at cutoff_date. realized_dates, if not
        given, defaults to the same business-day index as the simulation
        (use the actual realized index instead when lengths can differ).
        realized_nav must start at V0 on day 0 (the cutoff) if provided.
        """
        if self.ptf_nav is None:
            self.simulate()
        nav = self.ptf_nav
        n = nav.shape[0]
        if self.cutoff_date is not None:
            x = pd.bdate_range(start=self.cutoff_date, periods=n)
        else:
            x = np.arange(n)
        p = {q: np.percentile(nav, q, axis=1) for q in (5, 25, 50, 75, 95)}

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.fill_between(x, p[5], p[95], color="steelblue", alpha=0.18, label="p5 to p95")
        ax.fill_between(x, p[25], p[75], color="steelblue", alpha=0.35, label="p25 to p75")
        ax.plot(x, p[50], color="navy", lw=1.5, label="simulated median")
        if realized_nav is not None:
            if realized_dates is not None:
                rx = realized_dates
            elif self.cutoff_date is not None:
                rx = pd.bdate_range(start=self.cutoff_date, periods=len(realized_nav))
            else:
                rx = np.arange(len(realized_nav))
            ax.plot(rx, realized_nav, color="red", lw=2.0, label="realized")
        ax.axvline(pd.Timestamp("2025-04-02"), color="black", ls="--", lw=1.2)
        ax.set_title(f"Simulated NAV distribution vs realized ({self.profile}, "
                     f"horizon={self.th} days)")
        ax.set_xlabel("date" if self.cutoff_date is not None else "days from cutoff")
        ax.set_ylabel("NAV")
        ax.legend(fontsize=9)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(fname, dpi=140)
        plt.close(fig)

    def plot_mdd_distribution(self, fname="mc_mdd_distribution.png"):
        """(1)+(2) Portfolio MDD histogram with percentiles and the zero-drift reference."""
        if self.summary is None:
            self.analyse()
        mdd = np.abs(self.ptf_mdd)
        sigma_ptf = float(np.sqrt(self.weights @ self.cov_mat @ self.weights))

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.hist(mdd, bins=60, color="steelblue", alpha=0.8)
        for q, c in [(5, "grey"), (50, "black"), (95, "firebrick")]:
            v = np.percentile(mdd, q)
            ax.axvline(v, color=c, ls="--", lw=1.2, label=f"p{q} = {v:.2%}")
        ref = self.analytical_emd_zero_drift(sigma_ptf)
        ax.axvline(ref, color="green", lw=1.5, label=f"zero-drift ref = {ref:.2%}")
        ax.set_title(f"Portfolio MDD distribution vs analytical reference ({self.profile})")
        ax.set_xlabel("Maximum drawdown (magnitude)"); ax.set_ylabel("paths")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(fname, dpi=140)
        plt.close(fig)

    def plot_depth_vs_duration(self, fname="mc_depth_vs_duration.png"):
        """(3) Drawdown depth vs peak-to-trough duration, one point per path."""
        if self.ptf_mdd is None:
            self.analyse()
        mdd = np.abs(self.ptf_mdd)
        ps_stats = self._drawdown_stats(self.ptf_nav)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.scatter(ps_stats["peak_to_trough"], mdd, s=6, alpha=0.25, color="darkorange")
        ax.set_title(f"Depth vs peak-to-trough duration ({self.profile})")
        ax.set_xlabel("peak to trough (days)"); ax.set_ylabel("MDD magnitude")
        fig.tight_layout()
        fig.savefig(fname, dpi=140)
        plt.close(fig)

    def plot_convergence(self, fname="mc_convergence.png"):
        """(4) Running mean MDD as the number of paths accumulates."""
        if self.ptf_mdd is None:
            self.analyse()
        mdd = np.abs(self.ptf_mdd)
        ns, run = self.convergence()

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.plot(ns, run, color="purple")
        ax.axhline(mdd.mean(), color="black", ls=":", lw=1, label=f"final mean = {mdd.mean():.2%}")
        ax.set_title(f"Convergence of mean MDD ({self.profile})")
        ax.set_xlabel("number of paths"); ax.set_ylabel("running mean MDD")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(fname, dpi=140)
        plt.close(fig)

    def plot_sensitivity(self, fname="mc_sensitivity.png"):
        """(5) Sensitivity of mean MDD to mu and sigma scaling."""
        ss, ms, grid = self.sensitivity()

        fig, ax = plt.subplots(figsize=(8, 5.5))
        for i, m in enumerate(ms):
            ax.plot(ss, grid[i], marker="o", label=f"mu x {m:g}")
        ax.axvline(1.0, color="grey", ls=":", lw=1)
        ax.set_title(f"Sensitivity of mean MDD to mu and sigma ({self.profile})")
        ax.set_xlabel("sigma scale (1.0 = estimate)"); ax.set_ylabel("mean MDD magnitude")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(fname, dpi=140)
        plt.close(fig)

    def plot_portfolio(self, prefix="mc_portfolio"):
        """Convenience wrapper: saves the four diagnostics above as separate files."""
        self.plot_mdd_distribution(f"{prefix}_mdd_distribution.png")
        self.plot_depth_vs_duration(f"{prefix}_depth_vs_duration.png")
        self.plot_convergence(f"{prefix}_convergence.png")
        self.plot_sensitivity(f"{prefix}_sensitivity.png")


    def plot_assets(self, fname):
        """Cross sectional bar of mean MDD: portfolio vs each asset, with zero-drift ref."""
        if self.summary is None:
            self.analyse()
        s = self.summary
        x = np.arange(len(s))
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.bar(x, s["mean_MDD"], color="steelblue", alpha=0.8, label="mean MDD")
        yerr = np.vstack([s["mean_MDD"] - s["median_MDD"], s["p95_MDD"] - s["mean_MDD"]])
        ax.errorbar(x, s["mean_MDD"], yerr=np.abs(yerr), fmt="none",
                    ecolor="black", capsize=5, label="median to p95")
        ax.plot(x, s["analytical_EMD_zero_drift"], "D", color="green",
                label="zero-drift analytical ref")
        ax.set_xticks(x); ax.set_xticklabels(s.index, rotation=20, ha="right")
        ax.set_ylabel("Maximum drawdown (magnitude)")
        ax.set_title(f"Expected MDD: portfolio vs assets ({self.profile}, T={self.th})")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(fname, dpi=140)
        plt.close(fig)