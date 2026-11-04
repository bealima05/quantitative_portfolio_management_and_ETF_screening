import pandas as pd
import yfinance as yf
import numpy as np

from datetime import datetime, timedelta

class Data:
    """
    Colects input data and returns it treated so it can be analised.

    Input:
    ----------- 
    file: inout file
    etf_col: etf column name
    """

    def __init__ (self, file, prf, etf_col, bm_col, class_col, rf_col, start_date, end_date):
        self.file = file
        self.prf = prf
        self.etf_col = etf_col
        self.bm_col = bm_col
        self.class_col = class_col
        self.rf_col = rf_col
        self.start_date = start_date
        self.end_date = end_date

        self.df = pd.read_excel(file).sort_values(etf_col)

        self.df_profile = None
        self.df_bm_profile = None

        # Class allocation
        self.cl_alloc = None

        # Lists with the ETFs and respectively benchmarks
        self.etf_list = None
        self.bm_list = None

        # Dataframe with ETFs historical prices: open and close
        self.df_open = None
        self.df_close = None

        # Dataframe with BMs historical prices: open and close
        self.df_bm_open = None
        self.df_bm_close = None

        # Dataframe with ETFs close returns
        self.df_returns = None
        self.df_simple_returns = None

        # Dataframe with BMs close returns
        self.df_bm_returns = None

        # Dataframe with risk free portfolio
        self.df_rf = None

        # Dataframe with theoretical portfolios returns
        self.theo_ret = None
        self.theo_vol = None

    def profile_df(self):
        profile_cols = [col for col in self.df.columns if col.upper().startswith("DC")]
        self.df_profile = self.df.set_index(self.etf_col)[profile_cols]
        self.df_bm_profile = self.df.groupby(self.bm_col)[profile_cols].sum()


    def class_allocation(self):
        classes = self.df.set_index(self.etf_col)[self.class_col]
        self.cl_alloc = self.df_profile.groupby(classes).sum()


    def extract_data(self):

        # 1. Isolate the active ETFs for the given profile (weight > 0)
        profile_weights = self.df_profile[self.prf]
        self.etf_list = profile_weights[profile_weights > 0].index.tolist()

        # 2. Extract corresponding benchmarks for ONLY those active ETFs
        mask = self.df[self.etf_col].isin(self.etf_list)
        self.bm_list = self.df.loc[mask, self.bm_col].dropna().unique().tolist()

        # 3. Extract ETF data
        data = yf.download(self.etf_list, start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)
        valid_dates = data[["Open", "Close"]].dropna(how="any").index

        self.df_open = data["Open"].loc[valid_dates]
        self.df_close = data["Close"].loc[valid_dates]

        # 4. Extract BM data
        bm_data = yf.download(self.bm_list, start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)
        bm_valid_dates = bm_data[["Open", "Close"]].dropna(how="any").index

        self.df_bm_open = bm_data["Open"].loc[bm_valid_dates]
        self.df_bm_close = bm_data["Close"].loc[bm_valid_dates]


    def asset_returns(self):
        self.df_returns = np.log(self.df_close/ self.df_close.shift(1)).dropna(how="all")
        self.df_bm_returns = np.log(self.df_bm_close/self.df_bm_close.shift(1)).dropna(how="all")

        self.df_simple_returns = self.df_close.pct_change().dropna(how="all")


    def risk_free(self):
        self.df_rf = self.df.set_index(self.etf_col)[self.rf_col]


    def theoric(self):
        """Simulates a portfolio that is rebalanced every day just for theoretical purposes"""

        # 1. Grab the actual pandas Series of weights using the column name (self.prf)
        weights = self.df_profile[self.prf]
        
        # 2. Filter for weights greater than 0
        active_weights = weights[weights > 0]
        
        # 3. Align returns dataframe with the weights index. 
        # yfinance sorts alphabetically, so if we don't align, pandas .dot() might misalign or return NaNs!
        aligned_returns = self.df_returns[active_weights.index]
        
        # 4. Calculate daily portfolio returns and annualized volatility
        self.theo_ret = aligned_returns.dot(active_weights)
        self.theo_vol = self.theo_ret.std() * np.sqrt(252)




    

        


