import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import norm
from datetime import date

date_today = date.today()

class ExPost:
    """
    Esta classe está focada em criar métricas de avaliação das carteiras com base no método ex post, 
    isto é, com base em eventos conhecidos, de modo a avaliar a performance real do portfólio.
    """

    def __init__(self, df_profile, profile, df_returns, df_rf, ptf_ret, ptf_nav, bm_ret, **kwargs):

        self.df_profile = df_profile
        self.profile = profile
        self.df_ret = df_returns
        self.df_rf = df_rf
        self.ptf_nav = ptf_nav
        self.ptf_ret = ptf_ret
        self.bm_ret = bm_ret


        # Dinamic Parameters
        self.vol_lookback = kwargs.get('vol_lookback', 21)
        
        self.prf = None
        self.rf_ptf_ret = None
        

        # Drawdown
        self.mdd = None
        self.mdd_index = None
        self.dd_series = None

        # Mean
        self.mean = None
        
        # Variance
        self.variance = None

        # Covariance 
        self.cov = None
        
        # Correlation
        self.corr = None
        self.axis_corr = None

        # Volatility
        self.ptf_ann_vol = None
        self.roll_ptf_ann_vol = None

        # Excess Returns
        self.ptf_er = None
        self.bm_er = None

        # Alpha and Beta
        self.alpha = None
        self.beta = None

        # Tracking Error 
        self.te_ann = None

        self.er_rf = None
        self.er_rf_bm = None

        # Estimation window end date for ex-ante vol/VaR/CVaR
        self.train_end = pd.Timestamp(kwargs.get('train_end', '2024-12-31'))

        # Confidence level
        self.cl = kwargs.get('confidence_level', 0.95)

        # Estimation-window volatility (used for VaR/CVaR)
        self.est_ann_vol = None

        # Value at Risk
        self.var = None

        # Conditional Value at Risk
        self.cvar = None

        self.V0 = kwargs.get('init_cap',10000)


        


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
        self.er_rf_bm = self.bm_ret - self.rf_ptf_ret
    

    def excess_returns(self):
        """
        Calcula os retornos em excesso: Retorno do Portfólio (Série Temporal) - Taxa Risk-Free (Escalar).
        """
        # ETFs portfolio excess return
        self.er = self.ptf_ret - self.bm_ret
        self.er_ann_mean = np.mean(self.er)*252


    #def info_ratio(self):

        #self.er_info_ratio = np.mean(self.er)/np.std(self.er) *np.sqrt(252)

    
    def max_drawdown(self):
        """
        Computes the maximum drawdown the temporal series of drawdowns.
        """

        max_nav = self.ptf_nav.cummax()

        self.dd_series = (self.ptf_nav/max_nav) - 1.0

        self.mdd = self.dd_series.min()

        self.mdd_index = self.dd_series.idxmin().strftime('%Y-%m-%d')

   
    def mean(self):
        self.mean = self.ptf_ret.mean()


    def variance(self):
        self.variance = self.ptf_ret.var()
    

    def covariance_matrix(self):

        self.cov = self.df_ret.cov()

    
    def correlation_matrix(self):

        # Correlation Matrix
        self.corr = self.df_ret.corr()

        # Correlation heatmap
        self.axis_corr = sns.heatmap( 
        self.corr,
        vmin=-1, vmax=1, center=0,
        cmap="coolwarm",
        square=True
        )

        
    def annualized_vol(self):
        """Computes ETFs portfolio annualized volatility"""

        ptf_vol = self.ptf_ret.std()

        self.ptf_ann_vol = ptf_vol * np.sqrt(252)
    

    def rolling_annualized_vol(self):
        """Computes ETFs portfolio rolling annualized volatility using a rolling window of 21 days"""

        ptf_vol = self.ptf_ret.rolling(window=self.vol_lookback).std()

        self.roll_ptf_ann_vol = ptf_vol * np.sqrt(252)

    
    def beta_alpha(self):
        """
        Calculates Alpha and Beta using statsmodels OLS linear regression.
        Formula: Portfolio_Excess_Return = Alpha + Beta * Benchmark_Excess_Return
        """
        # 1. Ensure excess returns are calculated
        if self.er_rf is None or self.er_rf_bm is None:
            self.excess_returns_rf()
        # 3. Align data into a temporary DataFrame to handle any matching/missing dates
        df_ols = pd.concat([self.er_rf, self.er_rf_bm], axis=1).dropna()
        df_ols.columns = ['ptf_ret', 'bm_ret']
        if len(df_ols) < 2:
            self.alpha, self.beta = 0.0, 0.0
            print("Warning: Insufficient data to calculate Beta and Alpha.")
            return
        # 4. Set up the Target (y) and Predictor (X)
        y = df_ols['ptf_ret']
        X = df_ols['bm_ret']
        # CRITICAL STEP: statsmodels needs you to manually add a column of 1s 
        # so it can calculate the intercept (Alpha)
        X = sm.add_constant(X)
        # 5. Fit the OLS model
        model = sm.OLS(y, X).fit()
        # 6. Extract Alpha (intercept) and Beta (slope)
        # statsmodels names the constant column 'const' by default
        self.alpha = model.params['const']
        self.beta = model.params['bm_ret']
        
        # 7. Save the error term (residuals) for tracking error / idiosyncratic risk calculations
        self.epsilon = model.resid

    
    def tracking_error(self): 
        self.te_ann = (self.ptf_ret-self.bm_ret).std() * np.sqrt(252)



    




    

    




        


        
