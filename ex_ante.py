import pandas as pd
import numpy as np

from scipy.stats import norm


class ExAnte:
    """
    This class focuses on creating evaluation metrics for portfolios based on the ex-ante method,
    i.e., making forecasts of portfolio performance.
    """

    def __init__(self, df_profile, profile, ptf_state_ledger, bm_state_ledger, df_returns, df_rf, ptf_ret, ptf_nav, bm_ret, V0, **kwargs):

        self.df_profile = df_profile
        self.profile = profile
        self.ptf_sl = ptf_state_ledger
        self.bm_sl = bm_state_ledger
        self.df_ret = df_returns
        self.df_rf = df_rf
        self.ptf_ret = ptf_ret
        self.ptf_nav = ptf_nav
        self.bm_ret = bm_ret
        self.V0 = V0

        # Confidence level
        self.cl = kwargs.get('confidence_level', 0.95)
        
        # Look back volatility window
        self.vol_lookback = kwargs.get('vol_lookback', 21)

        # Time horizon
        self.th = kwargs.get('time_horizon',252)
        

        self.prf = None
        self.rf_ptf = None

        # Excess Returns
        self.er_rf = None
        self.er_rf_mean = None
        self.er_rf_std = None
        
        # Volatility
        self.ptf_ann_vol = None
        self.roll_ptf_ann_vol = None

        # Alpha and Beta
        self.beta = None

        # Estimation window end date for ex-ante vol/VaR/CVaR
        self.train_end = pd.Timestamp(kwargs.get('train_end', '2024-12-31'))

        # Estimation-window volatility (used for VaR/CVaR)
        self.est_ann_vol = None

        # Value at Risk
        self.var = None

        # Conditional Value at Risk
        self.cvar = None

        # tracking erro
        self.te = None



    def get_profile(self):
        self.prf = self.df_profile[self.profile]
    


    def risk_free(self):
        """
        Constrói o portfólio risk-free para o perfil instanciado.
        """

        if self.prf is None:
            self.get_profile()

        # Returns
        self.rf_ptf_ret = self.df_rf.dot(self.prf)

    def excess_returns_rf(self):
        """
        Calcula os retornos em excesso: Retorno do Portfólio (Série Temporal) - Taxa Risk-Free (Escalar).
        """
        
        if self.rf_ptf_ret is None:
            self.risk_free()
  
        # ETFs portfolio excess return
        self.er_rf = self.ptf_ret - self.rf_ptf_ret
        self.er_rf_mean = self.er_rf.mean()
        self.er_rf_std = self.er_rf.std()


    def sharpe_ratio(self):
        """Computes ETFs portfolio sharpe ratio"""

        self.sr = self.er_rf_mean / self.er_rf_std * np.sqrt(252)
    

    def excess_returns(self):
        """
        Calculates excess returns: Portfolio Return (Time Series) - Risk-Free Rate (Escalar).
        """

        ptf_returns = self.ptf_sl['ea_ptf_ret']
        bm_returns = self.bm_sl['ea_ptf_ret']
        self.er = ptf_returns - bm_returns
        self.er_mean = self.er.mean()
        self.er_std = self.er.std()


    
    def annualized_vol(self):
        """Computes ETFs portfolio ex-ante annualized volatility"""

        ptf_vol = self.ptf_ret.std()

        self.ptf_ann_vol = ptf_vol * np.sqrt(252)



    def beta_(self):
        """Computes ex ante Beta using the formula:
        beta = Cov(R_etf, R_bm)/Var(R_bm)"""

        self.beta = self.ptf_ret.cov(self.bm_ret)/self.bm_ret.var()



    def estimation_vol(self):
        """
        Computes annualized portfolio volatility using only the estimation
        window (start → train_end, default 31-12-2024). This is the correct
        ex-ante vol: estimated on past data, used to forecast forward risk.
        """
        train_ret = self.ptf_ret.loc[:self.train_end]
        self.est_ann_vol = train_ret.std() * np.sqrt(252)


    def value_at_risk(self):
        """
        Parametric VaR using estimation-window volatility.
        Calls estimation_vol() automatically if not yet computed.
        """
        if self.est_ann_vol is None:
            self.estimation_vol()

        z = norm.ppf(self.cl)
        self.var = self.est_ann_vol * z * self.V0

        #print(f'Portfolio VaR corresponding to 0.95 confidence level is {self.var:.3f}€')


    def conditional_value_at_risk(self):
        """
        Historical CVaR using only the estimation window (start → train_end).
        """
        train_ret = self.ptf_ret.loc[:self.train_end]

        df = pd.DataFrame()
        df['portfolio'] = self.V0 * (1 + train_ret).cumprod()
        df['losses'] = self.V0 - df['portfolio']

        var_zero = self.V0 - df['portfolio'].quantile(0.05)

        self.cvar = df.loc[df['losses'] > var_zero, 'losses'].mean()

        #print(f'Portfolio CVaR corresponding to 0.95 confidence level is {self.cvar:.3f}€')

        

    def tracking_error(self):

        # mean projection
        new_mean = self.th * self.df_ret.mean()

        cov = self.df_ret.cov()

        # covariance projection
        new_cov = self.th * cov

        # expected linear returns
        m_n = np.exp(new_mean + 0.5 * np.diag(new_cov)) - 1

        S = np.outer(m_n + 1, m_n + 1) * (np.exp(new_cov) - 1)

        # Extract the Pandas Series for the weights
        ef_series = self.ptf_sl.filter(regex='^Weight_').iloc[-1]
        eb_series = self.bm_sl.filter(regex='^Weight_').iloc[-1]
        
        ef = ef_series.to_numpy()
        eb = eb_series.to_numpy()
       
        ea = ef-eb

        # Convert S to pure NumPy arrays to avoid Pandas index mismatch during multiplication
        S_array = S.to_numpy() 

        # Calculate final ex-ante tracking error
        self.te = np.sqrt(ea @ S_array @ ea)



    def risk_contribution(self):
        """
        Computes each asset's marginal and component contribution to the
        ex-ante portfolio risk, using the current portfolio weights and the
        annualized return covariance matrix.

        MCR_i = (Sigma w)_i / sigma_p
        CCR_i = w_i * MCR_i          (component contribution, sums to sigma_p)
        """
        # Annualized covariance from daily log-returns
        cov = self.df_ret.cov() * 252
        cov_array = cov.to_numpy()

        # Current portfolio weights (latest row of the state ledger)
        w_series = self.ptf_sl.filter(regex='^Weight_').iloc[-1]
        w = w_series.to_numpy()

        # Portfolio volatility
        sigma_p = np.sqrt(w @ cov_array @ w)

        # Marginal and component contributions
        self.mcr = (cov_array @ w) / sigma_p
        self.ccr = w * self.mcr
        self.pct_ccr = self.ccr / sigma_p

        # Index by asset name for readability
        self.mcr = pd.Series(self.mcr, index=w_series.index)
        self.ccr = pd.Series(self.ccr, index=w_series.index)
        self.pct_ccr = pd.Series(self.pct_ccr, index=w_series.index)