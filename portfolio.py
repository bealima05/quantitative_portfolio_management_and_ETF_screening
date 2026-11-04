import pandas as pd
import numpy as np
import os
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class Portfolio:
    """
    Class responsible for the creation, simulation and active management of the portfolio.
    Allows for daily incrementation and update of excel memory.
    """

    def __init__(self, df_profile, df_bm_close, df_close, df_open, df_returns, df_rf, profile, method, V0,  **kwargs):
        
        self.df_profile = df_profile
        self.profile = profile
        self.df_bm_close = df_bm_close
        self.df_close = df_close
        self.df_open = df_open
        self.df_returns = df_returns
        self.df_rf = df_rf
        self.method = method  
        self.V0 = V0

        self.prf = None
        self.df_init_units = None
        self.rf_ptf = None
        
        self.state_ledger = None 
        self.df_execution_log = None
        self.df_estimation_log = None
        self.is_dirty = False
        
        # =====================================================================
        # Dinamic Parameters
        # =====================================================================
        self.vol_lookback = kwargs.get('vol_lookback', 21)
        self.rebalance_period = kwargs.get('rebalance_period', 21)
        self.epsilon = kwargs.get('epsilon', 1e-6)
        self.bps = kwargs.get('bps', 0.0005)
        self.drift_band = kwargs.get('drift_band', 0.0009)
        self.start_date = kwargs.get('start_date')


    def get_profile(self):
        self.prf = self.df_profile[self.profile]
        #profile_weights = self.df_profile[self.prf]
        self.etf_list = self.prf[self.prf > 0].index.tolist()



    def _init_units(self):
        self.init_etf_cap = (self.V0 * self.prf)/(1+self.bps)
        init_prices = self.df_open.iloc[0]
        self.df_init_units = self.init_etf_cap.div(init_prices, axis=0)


        # Calculate exact capital deployed and initial fees at Open T=0
        clean_units = self.df_init_units.reindex(self.df_close.columns).fillna(0)
        clean_prices = init_prices.reindex(self.df_close.columns).fillna(0)        
        self.initial_investment = (clean_units * clean_prices).sum()
        initial_fees = self.initial_investment * self.bps
  
        # Store the remaining cash after day 0 execution
        self.initial_needed_cash = self.initial_investment + initial_fees
        self.initial_cash = round(self.V0 - self.initial_investment - initial_fees, 2)



    # =========================================================================
    # PORTFOLIO CONSTRUCTION METHODS
    # =========================================================================
        
    
    def _buy_hold(self):
        """ Buy and Hold """

        print("Construction of portfolio using Buy and Hold method")

        clean_units = self.df_init_units.reindex(self.df_close.columns).fillna(0)
        df_units = pd.DataFrame([clean_units.values] * len(self.df_close), index=self.df_close.index, columns=self.df_close.columns)

        cash_account = pd.Series(self.initial_cash, index=self.df_close.index)
        
        # EX POST
        df_values = df_units * self.df_close
        ser_tot_value = df_values.sum(axis=1) 
        df_weights = df_values.div(ser_tot_value, axis=0)
        df_ep_ptf_ret = ((df_weights * self.df_returns).sum(axis=1)).rename('ep_ptf_ret')
        df_cum_ret = self.initial_investment * (1 + df_ep_ptf_ret).cumprod()
        ser_nav = ser_tot_value + cash_account
        
        # EX ANTE 
        window_size = 21 
        df_mu = self.df_returns.rolling(window=window_size).mean().shift(1)
        df_ea_ptf_ret = (df_weights.shift(1) * df_mu).sum(axis=1).to_frame(name="ea_ptf_ret")

        # PREFIXING & CONVERTING TO DATAFRAMES FOR CONCATENATION
        df_op_prefixed = self.df_open.add_prefix('Open_')
        df_cp_prefixed = self.df_close.add_prefix('Close_')
        df_units_prefixed = df_units.add_prefix('Units_')
        df_weights_prefixed = df_weights.add_prefix('Weight_')
        
        # Convert your Series to single-column DataFrames here
        df_cash_col = cash_account.to_frame(name="Cash")
        df_cum_ret_col = df_cum_ret.to_frame(name="Cumulative_Return")
        df_nav_col = ser_nav.to_frame(name="NAV_close")


        self.state_ledger = pd.concat(
            [df_cash_col, df_nav_col, df_cum_ret_col, df_ep_ptf_ret, df_ea_ptf_ret, df_op_prefixed, df_cp_prefixed, df_units_prefixed, df_weights_prefixed], 
            axis=1
        )
        self.state_ledger.index.name = 'Data'
        self.is_dirty = True



    def _naive_risk_parity(self):
        """ Naive Risk Parity """

        print(f"Constructiong portfolio via Naive Risk Parity (Lookback: {self.vol_lookback}, Rebalance: {self.rebalance_period} days)...")

        df_vol = self.df_returns.rolling(window=self.vol_lookback).std().shift(1)
        safe_vol = df_vol.clip(lower=self.epsilon)
        inv_vol = 1.0 / safe_vol
        active_assets_mask = (self.prf > 0).astype(int)
        inv_vol_adjusted = inv_vol * active_assets_mask
        theoretical_weights = inv_vol_adjusted.div(inv_vol_adjusted.sum(axis=1), axis=0).fillna(0)

        valid_mask = df_vol.notna().all(axis=1)
        if not valid_mask.any():
            raise ValueError("Insuficient data to compute volatility")
        fst_day = valid_mask.idxmax()
        fst_idx = self.df_close.index.get_loc(fst_day)

        df_units = pd.DataFrame(np.nan, index=self.df_close.index, columns=self.df_close.columns)
        df_units.iloc[0] = self.df_init_units[self.df_close.columns]

        cash_account = pd.Series(np.nan, index=self.df_close.index)
        cash_account.iloc[0] = self.initial_cash

        evaluation_indices = list(range(fst_idx, len(self.df_close), self.rebalance_period))
        evaluation_dates = self.df_close.index[evaluation_indices]

        # Rebalancing motor
        current_units = df_units.iloc[0]
        cash_balance = self.initial_cash
        est_cash_balance = self.initial_cash

        execution_log = []
        estimation_log = []

        for date in evaluation_dates:

            cp = self.df_close.loc[date]
            etf_close_value = cp * current_units
            tot_val_close = etf_close_value.sum()
            est_nav= tot_val_close + cash_balance

            target_weights = theoretical_weights.loc[date]
            target_values = tot_val_close * target_weights

            # ESTIMATION LEDGER (Hypothetical execution at Close T) 
            est_new_units = target_values / cp
            est_units_trade = est_new_units - current_units
            
            is_est_sell = est_units_trade < -1e-6
            is_est_buy = est_units_trade > 1e-6

            est_cash_sell = (abs(est_units_trade[is_est_sell]) * cp[is_est_sell]).sum()
            est_cash_buy = (abs(est_units_trade[is_est_buy]) * cp[is_est_buy]).sum()

            est_trade_values = abs(est_units_trade) * cp
            est_fee_per_asset = est_trade_values * self.bps
            est_trade_fees = est_fee_per_asset.sum()

            est_available_cash = est_cash_balance + est_cash_sell
            est_cash_required = est_cash_buy + est_trade_fees
            
            est_injected_capital = 0.0
            
            if est_available_cash >= est_cash_required:
                est_cash_balance = est_available_cash - est_cash_required
            else:
                est_shortfall = est_cash_required - est_available_cash
                est_injected_capital = est_shortfall
                est_cash_balance = -est_shortfall
            
            for asset in target_weights.index:
                action = 'Hold'
                if is_est_buy[asset]: action = 'Buy'
                elif is_est_sell[asset]: action = 'Sell'
                
                estimation_log.append({
                    'Decision_Date': date,
                    'est_nav': est_nav,
                    'Asset': asset,
                    'Target Weight': target_weights[asset],
                    'Estimated Action': action,
                    'Estimated Trade Units': est_units_trade[asset],
                    'Estimated Fees': est_fee_per_asset[asset],
                    'Est_Cash_Balance': est_cash_balance,
                    'Est_Capital_Injection' : est_injected_capital,
                })

            # --- EXECUTION ---
            current_idx = self.df_close.index.get_loc(date)
            if current_idx + 1 >= len(self.df_close.index):
                break
            execution_date = self.df_close.index[current_idx+1]
            op = self.df_open.loc[execution_date]

            new_units = target_values / op
            #new_units = est_new_units
            units_trade = new_units - current_units

            is_sell = units_trade < -1e-6
            is_buy = units_trade > 1e-6

            cash_sell = (abs(units_trade[is_sell]) * op[is_sell]).sum()
            cash_buy = (abs(units_trade[is_buy]) * op[is_buy]).sum()

            trade_values = abs(units_trade) * op
            fee_per_asset = (trade_values * self.bps)
            trade_fees = fee_per_asset.sum()

            available_cash = cash_balance + cash_sell
            cash_required = cash_buy + trade_fees
            
            injected_capital = 0.0
            
            if available_cash >= cash_required:
                cash_balance = available_cash - cash_required
            else:
                shortfall = cash_required - available_cash
                injected_capital = shortfall
                cash_balance = -shortfall

            post_trade_etf_values = new_units * op
            post_trade_tot_val=post_trade_etf_values.sum()
            post_trade_nav = post_trade_tot_val + cash_balance
            post_trade_weights = post_trade_etf_values / post_trade_nav
            
            
            for asset in target_weights.index:
                trade_qty = units_trade[asset]
                
                action = 'Hold'
                if is_buy[asset]: 
                    action = 'Buy'
                elif is_sell[asset]: 
                    action = 'Sell'

                execution_log.append({
                    'Decision_Date': date,
                    'Execution_Date': execution_date,
                    'Nav_post_trade' : post_trade_nav,
                    'Asset': asset,
                    'Target Weight': target_weights[asset],
                    'True Weight': post_trade_weights[asset],
                    'Action': action,
                    'Trade Units': trade_qty,
                    'Fees': fee_per_asset[asset],
                    'Cash_Balance': cash_balance,
                    'Capital_Injected': injected_capital,
                    'Units_After': new_units[asset]      
                })

            df_units.loc[execution_date] = new_units
            cash_account.loc[execution_date] = cash_balance 
            current_units = new_units

        df_units = df_units.ffill()
        cash_account = cash_account.ffill() 

        self.df_execution_log = pd.DataFrame(execution_log)
        self.df_estimation_log = pd.DataFrame(estimation_log)

        # STATE LEGER CONSTRUCTION
        # ex post
        df_values = df_units * self.df_close
        df_nav = (df_values.sum(axis=1) + cash_account).to_frame(name="NAV_close")
        df_weights = df_values.div(df_nav["NAV_close"], axis=0)
        df_ep_ptf_ret = df_nav.pct_change().fillna(0).rename(columns={"NAV_close": "ep_ptf_ret"})
        df_cum_ret = self.initial_investment*(1 + df_ep_ptf_ret["ep_ptf_ret"]).cumprod()
        df_cum_ret_col = df_cum_ret.to_frame(name="Cumulative_Return")

        # ex ante
        window_size = 21 
        df_mu = self.df_returns.rolling(window=window_size).mean().shift(1)
        df_ea_ptf_ret = (df_weights * df_mu).sum(axis=1).to_frame(name="ea_ptf_ret")
    
        # Dataframe
        df_cp_prefixed = self.df_close.add_prefix('Close_')
        df_units_prefixed = df_units.add_prefix('Units_')
        df_weights_prefixed = df_weights.add_prefix('Weight_')
        df_op_prefixed = self.df_open.loc[self.df_close.index].add_prefix('Open_')
        df_cash_col = cash_account.to_frame(name="Cash")

        self.state_ledger = pd.concat(
            [df_cash_col, df_nav, df_cum_ret_col, df_ep_ptf_ret, df_ea_ptf_ret, df_op_prefixed, df_cp_prefixed, df_units_prefixed, df_weights_prefixed], 
            axis=1
        )
        self.state_ledger.index.name = 'Data'
        self.is_dirty = True



    def _volatility_targeting(self):
        """ Volatility Targeting """

        print(f"Constructing portfolio via Volatility Targeting (Lookback: {self.vol_lookback}, Rebalance: {self.rebalance_period} days)...")

        max_leverage = 1.0

        if self.df_close.empty or self.df_close.shape[1] < 2:
            raise ValueError("There aren't suficient assets.")

        # Portfolio simulation to compute target volatility
        weights = self.prf.loc[self.df_returns.columns]
        sim_init_units = self.df_init_units[self.df_close.columns]
        sim_etf_value = self.df_bm_close * sim_init_units.values
        sim_tot_val = sim_etf_value.sum(axis=1)
        sim_returns = np.log(sim_tot_val/sim_tot_val.shift(1))
        raw_sim_vol = sim_returns.rolling(self.vol_lookback).std()
        sim_df_vol = raw_sim_vol.shift(1)
        sim_df_target_vol = raw_sim_vol.expanding().mean().shift(1)

        # Data lenght
        valid_mask = sim_df_vol.notna() & sim_df_target_vol.notna()
        if not valid_mask.any():
            raise ValueError("There isn't enough data to compute volatility")
        fst_day = valid_mask.idxmax()
        fst_idx = self.df_close.index.get_loc(fst_day)

        df_units = pd.DataFrame(np.nan, index=self.df_close.index, columns=self.df_close.columns)
        df_units.iloc[0] = self.df_init_units[self.df_close.columns]

        cash_account = pd.Series(np.nan, index=self.df_close.index)
        cash_account.iloc[0] = self.initial_cash

        evaluation_indices = list(range(fst_idx, len(self.df_close), self.rebalance_period))
        evaluation_dates = self.df_close.index[evaluation_indices]

        # REBALANCING MOTOR
        current_units = df_units.iloc[0]
        cash_balance = self.initial_cash
        est_cash_balance = self.initial_cash

        execution_log = []
        estimation_log = []

        for date in evaluation_dates:

            # EVALUATION
            cp = self.df_close.loc[date]
            etf_close_value = cp * current_units
            nav_close = etf_close_value.sum() + cash_balance
            
            cur_vol = sim_df_vol.loc[date]
            cur_target = sim_df_target_vol.loc[date]

            lev = cur_target / cur_vol
            cur_lev = min(lev,1.0)

            target_weights = weights * cur_lev
            target_values = nav_close * target_weights

            # --- ESTIMATION LEDGER ---
            est_new_units = target_values / cp
            est_units_trade = est_new_units - current_units
            
            is_est_sell = est_units_trade < -1e-6
            is_est_buy = est_units_trade > 1e-6

            est_cash_sell = (abs(est_units_trade[is_est_sell]) * cp[is_est_sell]).sum()
            est_cash_buy = (abs(est_units_trade[is_est_buy]) * cp[is_est_buy]).sum()

            est_trade_values = abs(est_units_trade) * cp
            est_fee_per_asset = est_trade_values * self.bps
            est_trade_fees = est_fee_per_asset.sum()

            est_available_cash = est_cash_balance + est_cash_sell
            est_cash_required = est_cash_buy + est_trade_fees
            
            est_injected_capital = 0.0

            if est_available_cash >= est_cash_required:
                est_cash_balance = est_available_cash - est_cash_required
            else:
                est_shortfall = est_cash_required - est_available_cash
                est_injected_capital = est_shortfall
                est_cash_balance = -est_shortfall
            
            for asset in target_weights.index:
                action = 'Hold'
                if is_est_buy[asset]: action = 'Buy'
                elif is_est_sell[asset]: action = 'Sell'
                
                estimation_log.append({
                    'Decision_Date': date,
                    'NAV_close': nav_close,
                    'Target Vol': cur_target,
                    'Current Vol': cur_vol,
                    'Asset': asset,
                    'Target Weight': target_weights[asset],
                    'Estimated Action': action,
                    'Estimated Trade Units': est_units_trade[asset],
                    'Estimated Fees': est_fee_per_asset[asset],
                    'Est_Cash_Balance': est_cash_balance,
                    'Est_Capital_Injection' : est_injected_capital
                })

            # --- EXECUTION ---
            current_idx = self.df_close.index.get_loc(date)
            if current_idx + 1 >= len(self.df_close.index):
                break
            execution_date = self.df_close.index[current_idx+1]
            op = self.df_open.loc[execution_date]
            
            new_units = target_values / op
            units_trade = new_units - current_units

            is_sell = units_trade < -1e-6
            is_buy = units_trade > 1e-6

            cash_sell = (abs(units_trade[is_sell]) * op[is_sell]).sum()
            cash_buy = (abs(units_trade[is_buy]) * op[is_buy]).sum()

            trade_values = abs(units_trade) * op
            fee_per_asset = (trade_values * self.bps)
            trade_fees = fee_per_asset.sum()

            available_cash = cash_balance + cash_sell
            cash_required = cash_buy + trade_fees
            
            injected_capital = 0.0
            
            if available_cash >= cash_required:
                cash_balance = available_cash - cash_required
            else:
                shortfall = cash_required - available_cash
                injected_capital = shortfall
                cash_balance = -shortfall
            
            post_trade_etf_values = new_units * op
            post_trade_nav = post_trade_etf_values.sum() + cash_balance
            post_trade_weights = post_trade_etf_values / post_trade_nav
            
            for asset in weights.index:
                trade_qty = units_trade[asset]
                
                action = 'Hold'
                if is_buy[asset]: 
                    action = 'Buy'
                elif is_sell[asset]: 
                    action = 'Sell'

                execution_log.append({
                    'Decision_Date': date,            
                    'Execution_Date': execution_date,
                    'Nav_post_trade' : post_trade_nav,
                    'Target Vol' : cur_target,
                    'Current Vol' : cur_vol,
                    'Asset': asset,
                    'Target Weight': target_weights[asset],
                    'True Weight': post_trade_weights[asset],
                    'Action': action,
                    'Trade Units': trade_qty,
                    'Fees': fee_per_asset[asset],
                    'Cash_Balance': cash_balance,
                    'Capital_Injected': injected_capital,
                    'Units_After': new_units[asset]      
                })

            df_units.loc[execution_date] = new_units
            cash_account.loc[execution_date] = cash_balance 
            current_units = new_units
        
        df_units = df_units.ffill()
        cash_account = cash_account.ffill()

        self.df_execution_log = pd.DataFrame(execution_log)
        self.df_estimation_log = pd.DataFrame(estimation_log) 

        # STATE LEDGER CONSTRUCTION
        df_values = df_units * self.df_close
        df_tot_val = (df_values.sum(axis=1)).to_frame(name="tot_val_close")
        df_nav = (df_tot_val + cash_account).to_frame(name="NAV_close")
        df_weights = df_values.div(df_nav["NAV_close"], axis=0)
        raw_vt_returns = df_nav['NAV_close'].pct_change()
        df_actual_vt_vol = raw_vt_returns.rolling(self.vol_lookback).std().shift(1)
        df_ep_ptf_ret = raw_vt_returns.fillna(0).to_frame(name="ep_ptf_ret")
        
        df_cum_ret = self.initial_investment*(1 + df_ep_ptf_ret["ep_ptf_ret"]).cumprod()
        df_cum_ret_col = df_cum_ret.to_frame(name="Cumulative_Return")

        window_size = 21 
        df_mu = self.df_returns.rolling(window=window_size).mean().shift(1)
        df_ea_ptf_ret = (df_weights * df_mu).sum(axis=1).to_frame(name="ea_ptf_ret")
        
        df_ref_vol_col = sim_df_vol.to_frame(name="Ref_Market_Vol")
        df_vt_vol_col = df_actual_vt_vol.to_frame(name="Actual_VT_Vol")
        df_target_col = sim_df_target_vol.to_frame(name="Target_Vol")
        df_cash_col = cash_account.to_frame(name="Cash")

        df_prices_prefixed = self.df_close.add_prefix('Close_')
        df_units_prefixed = df_units.add_prefix('Units_')
        df_weights_prefixed = df_weights.add_prefix('Weight_')
        df_op_prefixed = self.df_open.loc[self.df_close.index].add_prefix('Open_')

        self.state_ledger = pd.concat(
            [df_cash_col, df_nav, df_cum_ret_col, df_ep_ptf_ret, df_ea_ptf_ret, df_ref_vol_col, df_vt_vol_col, df_target_col,
             df_op_prefixed, df_prices_prefixed, df_units_prefixed, df_weights_prefixed], 
            axis=1
        )
        self.state_ledger.index.name = 'Data'
        self.is_dirty = True



    def _calendar_rebalancing(self):
        """ Calendar Rebalancing strategy"""

        print(f"Constructing portfolio via Calendar Rebalancing (Rebalancing each {self.rebalance_period} days)...")

        execution_log = []
        estimation_log = [] 

        target_weights = self.prf.loc[self.df_close.columns]

        df_units = pd.DataFrame(np.nan, index=self.df_close.index, columns=self.df_close.columns)
        df_units.iloc[0] = self.df_init_units[self.df_close.columns]
        
        cash_account = pd.Series(np.nan, index=self.df_close.index)
        cash_account.iloc[0] = self.initial_cash

        fst_idx = 0 
        evaluation_indices = list(range(fst_idx, len(self.df_close), self.rebalance_period))
        evaluation_dates = self.df_close.index[evaluation_indices]

        current_units = df_units.iloc[0]
        cash_balance = self.initial_cash
        est_cash_balance = self.initial_cash
            
        for date in evaluation_dates:
            if date not in self.df_close.index:
                continue

            cp = self.df_close.loc[date]
            etf_close_value = current_units * cp
            nav_close = etf_close_value.sum() + cash_balance
            target_values = nav_close * target_weights

            est_new_units = target_values / cp
            est_units_trade = est_new_units - current_units
            
            # ESTIMATION LEDGER 
            is_est_sell = est_units_trade < -1e-6
            is_est_buy = est_units_trade > 1e-6

            est_cash_sell = (abs(est_units_trade[is_est_sell]) * cp[is_est_sell]).sum()
            est_cash_buy = (abs(est_units_trade[is_est_buy]) * cp[is_est_buy]).sum()

            est_trade_values = abs(est_units_trade) * cp
            est_fee_per_asset = est_trade_values * self.bps
            est_trade_fees = est_fee_per_asset.sum()

            est_available_cash = est_cash_balance + est_cash_sell
            est_cash_required = est_cash_buy + est_trade_fees
            
            est_injected_capital = 0.0
            
            if est_available_cash >= est_cash_required:
                est_cash_balance = est_available_cash - est_cash_required
            else:
                est_shortfall = est_cash_required - est_available_cash
                est_injected_capital = est_shortfall
                est_cash_balance = -est_shortfall
            
            for asset in target_weights.index:
                action = 'Hold'
                if is_est_buy[asset]: action = 'Buy'
                elif is_est_sell[asset]: action = 'Sell'
                
                estimation_log.append({
                    'Decision_Date': date,
                    'NAV_close': nav_close,
                    'Asset': asset,
                    'Target Weight': target_weights[asset],
                    'Estimated Action': action,
                    'Estimated Trade Units': est_units_trade[asset],
                    'Estimated Fees': est_fee_per_asset[asset],
                    'Est_Cash_Balance': est_cash_balance,
                    'Est_Capital_Injection' : est_injected_capital,
                })

            # --- EXECUTION ---    
            current_idx = self.df_close.index.get_loc(date)
            if current_idx + 1 >= len(self.df_close.index):
                break 
                
            execution_date = self.df_close.index[current_idx + 1]
            op = self.df_open.loc[execution_date]
            
            new_units = target_values / op
            units_to_trade = new_units - current_units
            
            is_sell = units_to_trade < -1e-6
            is_buy = units_to_trade > 1e-6
            
            cash_from_sells = (abs(units_to_trade[is_sell]) * op[is_sell]).sum()
            cost_of_buys = (units_to_trade[is_buy] * op[is_buy]).sum()
            
            trade_values = abs(units_to_trade) * op
            fee_per_asset = trade_values * self.bps
            total_fees = fee_per_asset.sum()
            
            total_available_cash = cash_balance + cash_from_sells
            total_cash_required = cost_of_buys + total_fees
            
            injected_capital = 0.0
            
            if total_available_cash >= total_cash_required:
                cash_balance = total_available_cash - total_cash_required
            else:
                shortfall = total_cash_required - total_available_cash
                injected_capital = shortfall
                cash_balance = -shortfall 
            
            post_trade_etf_values = new_units * op
            post_trade_nav = post_trade_etf_values.sum() + cash_balance
            post_tot_val = post_trade_etf_values.sum()
            post_trade_weights = post_trade_etf_values / post_trade_nav
                
            for asset in target_weights.index:
                trade_qty = units_to_trade[asset]
                
                action = 'Hold'
                if is_buy[asset]: 
                    action = 'Buy'
                elif is_sell[asset]: 
                    action = 'Sell'
                    
                execution_log.append({
                    'Decision_Date': date,
                    'Execution_Date': execution_date,
                    'Nav_post_trade' : post_trade_nav,
                    'Asset': asset,
                    'Target Weight': target_weights[asset],
                    'True Weight': post_trade_weights[asset],
                    'Action': action,
                    'Trade Units': trade_qty,
                    'Fees': fee_per_asset[asset],
                    'Cash_Balance': cash_balance,
                    'Capital_Injected': injected_capital,
                    'Units_After': new_units[asset]      
                })
                
            df_units.loc[execution_date] = new_units
            cash_account.loc[execution_date] = cash_balance
            current_units = new_units.copy()

        df_units = df_units.ffill()
        cash_account = cash_account.ffill()

        self.df_execution_log = pd.DataFrame(execution_log)
        self.df_estimation_log = pd.DataFrame(estimation_log)

        # STATE LEDGER CONSTRUCTION
        df_values = df_units * self.df_close
        df_nav = (df_values.sum(axis=1) + cash_account).to_frame(name="NAV_close")
        df_weights = df_values.div(df_nav["NAV_close"], axis=0)
        df_ep_ptf_ret = df_nav.pct_change().fillna(0).rename(columns={"NAV_close": "ep_ptf_ret"})
        df_cum_ret = self.initial_investment*(1 + df_ep_ptf_ret["ep_ptf_ret"]).cumprod()
        df_cum_ret_col = df_cum_ret.to_frame(name="Cumulative_Return")

        # ex ante
        window_size = 21 
        df_mu = self.df_returns.rolling(window=window_size).mean().shift(1)
        df_ea_ptf_ret = (df_weights * df_mu).sum(axis=1).to_frame(name="ea_ptf_ret")

        # To dataframe
        df_cash_col = cash_account.to_frame(name="Cash")
        df_prices_prefixed = self.df_close.add_prefix('Close_')
        df_units_prefixed = df_units.add_prefix('Units_')
        df_weights_prefixed = df_weights.add_prefix('Weight_')
        df_op_prefixed = self.df_open.loc[self.df_close.index].add_prefix('Open_')

        self.state_ledger = pd.concat(
            [df_cash_col, df_nav, df_cum_ret_col, df_ep_ptf_ret, df_ea_ptf_ret, df_op_prefixed, df_prices_prefixed, df_units_prefixed, df_weights_prefixed], 
            axis=1
        )
        self.state_ledger.index.name = 'Data'
        self.is_dirty = True



    def _drift_rebalancing(self):
        """ Drift Rebalancing strategy"""
        
        execution_log = []
        estimation_log = []

        target_weights = self.prf.loc[self.df_close.columns]
    
        evaluation_dates = self.df_close.index 

        df_units = pd.DataFrame(np.nan, index=self.df_close.index, columns=self.df_close.columns)
        df_units.iloc[0] = self.df_init_units[self.df_close.columns]
        
        cash_account = pd.Series(np.nan, index=self.df_close.index)
        cash_account.iloc[0] = self.initial_cash

        current_units = df_units.iloc[0]
        cash_balance = self.initial_cash
        est_cash_balance = self.initial_cash
            
        for date in evaluation_dates:
            if date not in self.df_close.index:
                continue
                
            # EVALUATION
            cp = self.df_close.loc[date]
            etf_close_value = current_units * cp
            nav_close = etf_close_value.sum() + cash_balance
            
            actual_weights = etf_close_value / nav_close
            drift = actual_weights - target_weights
            
            if not (drift.abs() > self.drift_band).any():
                continue             
            
            # ESTIMATION LEDGER
            target_values = nav_close * target_weights 
            est_new_units = target_values / cp
            est_units_trade = est_new_units - current_units
            
            is_est_sell = est_units_trade < -1e-6
            is_est_buy = est_units_trade > 1e-6

            est_cash_sell = (abs(est_units_trade[is_est_sell]) * cp[is_est_sell]).sum()
            est_cash_buy = (abs(est_units_trade[is_est_buy]) * cp[is_est_buy]).sum()

            est_trade_values = abs(est_units_trade) * cp
            est_fee_per_asset = est_trade_values * self.bps
            est_trade_fees = est_fee_per_asset.sum()

            est_available_cash = est_cash_balance + est_cash_sell
            est_cash_required = est_cash_buy + est_trade_fees
            
            est_injected_capital = 0.0
            
            if est_available_cash >= est_cash_required:
                est_cash_balance = est_available_cash - est_cash_required
            else:
                est_shortfall = est_cash_required - est_available_cash
                est_injected_capital = est_shortfall
                est_cash_balance = -est_shortfall
            
            for asset in target_weights.index:
                action = 'Hold'
                if is_est_buy[asset]: action = 'Buy'
                elif is_est_sell[asset]: action = 'Sell'
                
                estimation_log.append({
                    'Decision_Date': date,
                    'NAV_close': nav_close,
                    'Asset': asset,
                    'Target Weight': target_weights[asset],
                    'Estimated Action': action,
                    'Estimated Trade Units': est_units_trade[asset],
                    'Estimated Fees': est_fee_per_asset[asset],
                    'Est_Cash_Balance': est_cash_balance,
                    'Est_Capital_Injection' : est_injected_capital,
                })
            
            
            # EXECUTION 
            current_idx = self.df_close.index.get_loc(date)
            if current_idx + 1 >= len(self.df_close.index):
                break
                            
            execution_date = self.df_close.index[current_idx + 1]
            op = self.df_open.loc[execution_date] 

            new_units = target_values / op
            units_to_trade = new_units - current_units
            
            is_sell = units_to_trade < -1e-6
            is_buy = units_to_trade > 1e-6
            
            cash_from_sells = (abs(units_to_trade[is_sell]) * op[is_sell]).sum()
            cost_of_buys = (units_to_trade[is_buy] * op[is_buy]).sum()
            
            trade_values = abs(units_to_trade) * op
            fee_per_asset = trade_values * self.bps
            total_fees = fee_per_asset.sum()
            
            total_available_cash = cash_balance + cash_from_sells
            total_cash_required = cost_of_buys + total_fees
            
            injected_capital = 0.0
            
            if total_available_cash >= total_cash_required:
                cash_balance = total_available_cash - total_cash_required
            else:
                shortfall = total_cash_required - total_available_cash
                injected_capital = shortfall
                cash_balance = -shortfall 
            
            post_trade_etf_values = new_units * op
            post_trade_nav = post_trade_etf_values.sum() + cash_balance
            post_trade_weights = post_trade_etf_values / post_trade_nav
                
            for asset in target_weights.index:
                trade_qty = units_to_trade[asset]
                
                action = 'Hold'
                if is_buy[asset]: 
                    action = 'Buy'
                elif is_sell[asset]: 
                    action = 'Sell'
                    
                execution_log.append({          
                    'Decision_Date': date,
                    'Execution_Date': execution_date,
                    'Nav_post_trade' : post_trade_nav,
                    'Asset': asset,
                    'Target Weight': target_weights[asset],
                    'True Weight': post_trade_weights[asset],
                    'Action': action,
                    'Trade Units': trade_qty,
                    'Fees': fee_per_asset[asset],
                    'Cash_Balance': cash_balance,
                    'Capital_Injected': injected_capital,
                    'Units_After': new_units[asset]      
                })
            
            df_units.loc[execution_date] = new_units
            cash_account.loc[execution_date] = cash_balance
            current_units = new_units.copy()

        df_units = df_units.ffill()
        cash_account = cash_account.ffill()

        self.df_execution_log = pd.DataFrame(execution_log)
        self.df_estimation_log = pd.DataFrame(estimation_log)

        # STATE LEDGER CONSTRUCTION
        df_values = df_units * self.df_close
        df_nav = (df_values.sum(axis=1) + cash_account).to_frame(name="NAV_close")
        df_weights = df_values.div(df_nav["NAV_close"], axis=0)
        df_ep_ptf_ret = df_nav.pct_change().fillna(0).rename(columns={"NAV_close": "ep_ptf_ret"})
        df_cum_ret = self.initial_investment*(1 + df_ep_ptf_ret["ep_ptf_ret"]).cumprod()
        df_cum_ret_col = df_cum_ret.to_frame(name="Cumulative_Return")

        # ex ante
        window_size = 21 
        df_mu = self.df_returns.rolling(window=window_size).mean().shift(1)
        df_ea_ptf_ret = (df_weights * df_mu).sum(axis=1).to_frame(name="ea_ptf_ret")

        # To dataframe
        df_cash_col = cash_account.to_frame(name="Cash")
        df_prices_prefixed = self.df_close.add_prefix('Close_')
        df_units_prefixed = df_units.add_prefix('Units_')
        df_weights_prefixed = df_weights.add_prefix('Weight_')
        df_op_prefixed = self.df_open.loc[self.df_close.index].add_prefix('Open_')

        self.state_ledger = pd.concat(
            [df_cash_col, df_nav, df_cum_ret_col, df_ep_ptf_ret, df_ea_ptf_ret, df_op_prefixed, df_prices_prefixed, df_units_prefixed, df_weights_prefixed], 
            axis=1
        )
        self.state_ledger.index.name = 'Data'
        self.is_dirty = True



    # =========================================================================
    # CENTRAL SELECTION
    # =========================================================================

    def build_portfolio(self):
        """
        Ativation of the chosen strategy.
        """
        self.get_profile()
        self._init_units()

        strategies = {
            "buy_hold": self._buy_hold,
            "naive_risk_parity": self._naive_risk_parity,
            "volatility_targeting": self._volatility_targeting,
            "calendar_rebalancing": self._calendar_rebalancing,
            "drift_rebalancing": self._drift_rebalancing
        }

        if self.method in strategies:
            strategies[self.method]()
        else:
            raise ValueError(f"Método '{self.method}' não reconhecido nas estratégias.")



    # =========================================================================
    # DAILY UPDATES
    # =========================================================================
    
    def append_daily_metrics(self, date, cp, op):
        """ Adds a the metrics of today to the state ledger and to the execution log """
        if self.state_ledger is None:
            raise ValueError("Portfolio history hasn't been built or downloaded.")
        
        if self.prf is None:
            self.get_profile()

        new_date = pd.to_datetime(date)

        if new_date in self.state_ledger.index:
            print(f"Warning: data for {new_date.date()} already exists on state ledger.")
            return

        last_row = self.state_ledger.iloc[-1]
        tickers = self.df_close.columns

        current_ref_vol = np.nan
        current_vt_vol = np.nan
        current_target = np.nan
        current_leverage = np.nan


        if self.method == "buy_hold":
            current_cash = last_row.get("Cash", self.initial_cash)
            current_units = pd.Series([last_row[f"Units_{tk}"] for tk in tickers], index=tickers)
            
            new_values = current_units * cp
            new_nav = new_values.sum() + current_cash
            new_weights = new_values / new_nav
            
            nav_yesterday = last_row["NAV_close"]
            ep_ptf_ret = (new_nav / nav_yesterday) - 1.0 if nav_yesterday > 0 else 0.0

            last_close_prices = pd.Series([last_row[f"Close_{tk}"] for tk in tickers], index=tickers)
            bod_values = current_units * last_close_prices
            bod_nav = bod_values.sum() + current_cash
            bod_weights = (bod_values / bod_nav).fillna(0)

            window_size = 21
            mu_vector = self.df_returns.iloc[-window_size:].mean() 
            ea_ptf_ret = (bod_weights * mu_vector).sum()
            

        elif self.method == "naive_risk_parity":
            base_weights = self.prf.loc[tickers]
            current_units = pd.Series([last_row[f"Units_{tk}"] for tk in tickers], index=tickers)
            current_cash = last_row.get("Cash", 0.0)
            bps = getattr(self, 'bps', 0.0005)
            
            units_cols = [f"Units_{tk}" for tk in tickers]
            df_units_history = self.state_ledger[units_cols]
            change = df_units_history.ne(df_units_history.shift(1)).any(axis=1)
            reb_days = change[change].index
            
            if len(reb_days) > 0:
                last_exec_day = reb_days[-1]
                days_since_exec = len(self.state_ledger.loc[last_exec_day:])
            else:
                days_since_exec = self.rebalance_period 

            if days_since_exec >= self.rebalance_period:
                print(f"   [!] NRP execution  (Evaluation: Yesterday | Execution: Today - at opening.")
                
                price_cols = [f"Close_{tk}" for tk in tickers]
                past_prices = self.state_ledger[price_cols].tail(self.vol_lookback + 1).copy()
                past_prices.columns = tickers 
                
                past_returns = np.log(past_prices / past_prices.shift(1)).dropna()
                current_vol_nrp = past_returns.std()
                
                safe_vol = np.clip(current_vol_nrp, a_min=self.epsilon, a_max=None)
                inv_vol = 1.0 / safe_vol
                
                active_assets_mask = (base_weights > 0).astype(int)
                inv_vol_adjusted = inv_vol * active_assets_mask
                
                target_weights = inv_vol_adjusted / inv_vol_adjusted.sum()
                target_weights = target_weights.fillna(0)
                
                yesterday_nav = last_row['NAV_close']
                target_values = yesterday_nav * target_weights

                # --- ESTIMATION LEDGER  ---
                yesterday_close = pd.Series([last_row[f"Close_{tk}"] for tk in tickers], index=tickers)
                est_new_units = target_values / yesterday_close
                est_units_trade = est_new_units - current_units
                
                is_est_sell = est_units_trade < -1e-6
                is_est_buy = est_units_trade > 1e-6

                est_cash_sell = (abs(est_units_trade[is_est_sell]) * yesterday_close[is_est_sell]).sum()
                est_cash_buy = (abs(est_units_trade[is_est_buy]) * yesterday_close[is_est_buy]).sum()

                est_trade_values = abs(est_units_trade) * yesterday_close
                est_fee_per_asset = est_trade_values * bps
                est_trade_fees = est_fee_per_asset.sum()

                est_available_cash = current_cash + est_cash_sell
                est_cash_required = est_cash_buy + est_trade_fees
                
                est_injected_capital = 0.0
                if est_available_cash >= est_cash_required:
                    est_cash_balance_end = est_available_cash - est_cash_required
                else:
                    est_shortfall = est_cash_required - est_available_cash
                    est_injected_capital = est_shortfall
                    est_cash_balance_end = -est_shortfall

                if getattr(self, 'df_estimation_log', None) is None:
                    self.df_estimation_log = pd.DataFrame()
                    
                daily_est_log = []
                for asset in target_weights.index:
                    action = 'Hold'
                    if is_est_buy[asset]: action = 'Buy'
                    elif is_est_sell[asset]: action = 'Sell'
                    
                    daily_est_log.append({
                        'Decision_Date': self.state_ledger.index[-1],
                        'NAV_close': yesterday_nav,
                        'Asset': asset,
                        'Target Weight': target_weights[asset],
                        'Estimated Action': action,
                        'Estimated Trade Units': est_units_trade[asset],
                        'Estimated Fees': est_fee_per_asset[asset],
                        'Est_Cash_Balance': est_cash_balance_end,
                        'Est_Capital_Injection': est_injected_capital
                    })
                
                if daily_est_log:
                    df_daily_est = pd.DataFrame(daily_est_log)
                    self.df_estimation_log = pd.concat([self.df_estimation_log, df_daily_est], ignore_index=True)
                
                # EXECUTION
                new_units = target_values / op
                units_trade = new_units - current_units
                
                is_sell = units_trade < -1e-6
                is_buy = units_trade > 1e-6

                cash_sell = (abs(units_trade[is_sell]) * op[is_sell]).sum()
                cash_buy = (abs(units_trade[is_buy]) * op[is_buy]).sum()

                trade_values = abs(units_trade) * op
                fee_per_asset = trade_values * bps
                trade_fees = fee_per_asset.sum()

                available_cash = current_cash + cash_sell
                cash_required = cash_buy + trade_fees
                
                injected_capital = 0.0
                
                if available_cash >= cash_required:
                    current_cash = available_cash - cash_required
                else:
                    shortfall = cash_required - available_cash
                    injected_capital = shortfall
                    current_cash = -shortfall
                
                post_trade_etf_values = new_units * op
                post_trade_nav = post_trade_etf_values.sum() + current_cash
                post_trade_weights = post_trade_etf_values / post_trade_nav
                
                if getattr(self, 'df_execution_log', None) is None:
                    self.df_execution_log = pd.DataFrame()
                    
                daily_log = []
                for asset in target_weights.index:
                    trade_qty = units_trade[asset]
                    action = 'Hold'
                    if is_buy[asset]: 
                        action = 'Buy'
                    elif is_sell[asset]: 
                        action = 'Sell'

                    daily_log.append({
                        'Decision_Date': self.state_ledger.index[-1],
                        'Execution_Date': new_date,                   
                        'Nav_post_trade' : post_trade_nav,
                        'Asset': asset,
                        'Target Weight': target_weights[asset],
                        'True Weight': post_trade_weights[asset],
                        'Action': action,
                        'Trade Units': trade_qty,
                        'Fees': fee_per_asset[asset],
                        'Cash_Balance': current_cash,
                        'Capital_Injected': injected_capital,
                        'Units_After': new_units[asset]
                    })
                
                if daily_log:
                    df_daily_log = pd.DataFrame(daily_log)
                    self.df_execution_log = pd.concat([self.df_execution_log, df_daily_log], ignore_index=True)

                current_units = new_units
            
            else:
                days_until = self.rebalance_period - days_since_exec
                print(f"   [-] It is not rebalancing day. Using Buy & Hold method (there are {days_until} days until rebalancing).")
                
            etf_value = (current_units * cp).sum()
            new_nav = etf_value + current_cash
            new_weights = (current_units * cp) / new_nav

            nav_yesterday = last_row["NAV_close"]
            ep_ptf_ret = (new_nav / nav_yesterday) - 1.0 if nav_yesterday > 0 else 0.0

            last_close_prices = pd.Series([last_row[f"Close_{tk}"] for tk in tickers], index=tickers)
            bod_values = current_units * last_close_prices
            bod_nav = bod_values.sum() + last_row.get("Cash", 0.0)
            bod_weights = (bod_values / bod_nav).fillna(0)

            window_size = 21
            mu_vector = self.df_returns.iloc[-window_size:].mean()
            ea_ptf_ret = (bod_weights * mu_vector).sum()
            
        
        elif self.method == "volatility_targeting":
            max_leverage = 2.0
            cash_ret = 0.0
            bps = getattr(self, 'bps', 0.0005)
            base_weights = self.prf.loc[tickers]
            
            current_units = pd.Series([last_row[f"Units_{tk}"] for tk in tickers], index=tickers)
            current_cash = last_row.get("Cash", 0.0)
            leverage_yesterday = last_row.get('Leverage', 1.0)
            
            if isinstance(cash_ret, pd.Series):
                cash_ret_today = cash_ret.loc[new_date]
            else:
                cash_ret_today = cash_ret

            current_cash = current_cash * (1 + cash_ret_today)
            
            units_cols = [f"Units_{tk}" for tk in tickers]
            df_units_history = self.state_ledger[units_cols]
            change = df_units_history.ne(df_units_history.shift(1)).any(axis=1)
            reb_days = change[change].index
            
            if len(reb_days) > 0:
                last_reb_day = reb_days[-1]
                days_since_reb = len(self.state_ledger.loc[last_reb_day:])
            else:
                days_since_reb = self.rebalance_period

            price_cols = [f"Close_{tk}" for tk in tickers]
            past_prices = self.state_ledger[price_cols].tail(self.vol_lookback).copy()
            past_prices.columns = tickers
            cp_row = pd.DataFrame([cp], index=[new_date])
            combined_prices = pd.concat([past_prices, cp_row], axis=0)
            
            base_init_units = self.df_init_units[tickers]
            base_nav = (combined_prices * base_init_units).sum(axis=1)
            base_returns = np.log(base_nav / base_nav.shift(1)).dropna()
            
            cur_vol = base_returns.std()
            historical_ref_vols = self.state_ledger['Ref_Market_Vol'].dropna()
            cur_target = pd.concat([historical_ref_vols, pd.Series([cur_vol])]).mean()

            if days_since_reb >= self.rebalance_period:
                print(f"   [!] Rebalancing by volatility target - there were {days_since_reb} days since the last rebalancing).")
                lev = cur_target / cur_vol
                current_leverage = min(lev, max_leverage)
                
                yesterday_nav = last_row['NAV_close']
                target_weights = base_weights * current_leverage
                target_allocations = yesterday_nav * base_weights * current_leverage

                # --- ESTIMATION LEDGER (Hypothetical execution at Yesterday's Close) ---
                yesterday_close = pd.Series([last_row[f"Close_{tk}"] for tk in tickers], index=tickers)
                est_new_units = target_allocations / yesterday_close
                est_units_trade = est_new_units - current_units
                
                is_est_sell = est_units_trade < -1e-6
                is_est_buy = est_units_trade > 1e-6

                est_cash_sell = (abs(est_units_trade[is_est_sell]) * yesterday_close[is_est_sell]).sum()
                est_cash_buy = (abs(est_units_trade[is_est_buy]) * yesterday_close[is_est_buy]).sum()

                est_trade_values = abs(est_units_trade) * yesterday_close
                est_fee_per_asset = est_trade_values * bps
                est_trade_fees = est_fee_per_asset.sum()

                est_available_cash = current_cash + est_cash_sell
                est_cash_required = est_cash_buy + est_trade_fees
                
                est_injected_capital = 0.0
                if est_available_cash >= est_cash_required:
                    est_cash_balance_end = est_available_cash - est_cash_required
                else:
                    est_shortfall = est_cash_required - est_available_cash
                    est_injected_capital = est_shortfall
                    est_cash_balance_end = -est_shortfall

                if getattr(self, 'df_estimation_log', None) is None:
                    self.df_estimation_log = pd.DataFrame()
                    
                daily_est_log = []
                for asset in target_weights.index:
                    action = 'Hold'
                    if is_est_buy[asset]: action = 'Buy'
                    elif is_est_sell[asset]: action = 'Sell'
                    
                    daily_est_log.append({
                        'Decision_Date': self.state_ledger.index[-1],
                        'NAV_close': yesterday_nav,
                        'Target Vol': cur_target,
                        'Current Vol': cur_vol,
                        'Asset': asset,
                        'Target Weight': target_weights[asset],
                        'Estimated Action': action,
                        'Estimated Trade Units': est_units_trade[asset],
                        'Estimated Fees': est_fee_per_asset[asset],
                        'Est_Cash_Balance': est_cash_balance_end,
                        'Est_Capital_Injection': est_injected_capital
                    })
                
                if daily_est_log:
                    df_daily_est = pd.DataFrame(daily_est_log)
                    self.df_estimation_log = pd.concat([self.df_estimation_log, df_daily_est], ignore_index=True)
                
                # --- EXECUTION ---
                new_units = target_allocations / op
                units_trade = new_units - current_units
                
                is_sell = units_trade < -1e-6
                is_buy = units_trade > 1e-6

                cash_sell = (abs(units_trade[is_sell]) * op[is_sell]).sum()
                cash_buy = (abs(units_trade[is_buy]) * op[is_buy]).sum()

                trade_values = abs(units_trade) * op
                fee_per_asset = trade_values * bps
                trade_fees = fee_per_asset.sum()

                available_cash = current_cash + cash_sell
                cash_required = cash_buy + trade_fees
                
                injected_capital = 0.0
                
                if available_cash >= cash_required:
                    current_cash = available_cash - cash_required
                else:
                    shortfall = cash_required - available_cash
                    injected_capital = shortfall
                    current_cash = -shortfall
                
                post_trade_etf_values = new_units * op
                post_trade_nav = post_trade_etf_values.sum() + current_cash
                post_trade_weights = post_trade_etf_values / post_trade_nav
                
                if getattr(self, 'df_execution_log', None) is None:
                    self.df_execution_log = pd.DataFrame()
                    
                daily_log = []
                for asset in base_weights.index:
                    trade_qty = units_trade[asset]
                    action = 'Hold'
                    if is_buy[asset]: 
                        action = 'Buy'
                    elif is_sell[asset]: 
                        action = 'Sell'
                    
                    daily_log.append({
                        'Decision_Date': self.state_ledger.index[-1],
                        'Execution_Date': new_date,
                        'Nav_post_trade' : post_trade_nav,
                        'Target Vol' : cur_target,
                        'Current Vol' : cur_vol,                  
                        'Asset': asset,
                        'Target Weight': target_weights[asset],
                        'True Weight': post_trade_weights[asset],
                        'Action': action,
                        'Trade Units': trade_qty,
                        'Fees': fee_per_asset[asset],
                        'Cash_Balance': current_cash,
                        'Capital_Injected': injected_capital,
                        'Units_After': new_units[asset]
                    })
                
                if daily_log:
                    df_daily_log = pd.DataFrame(daily_log)
                    self.df_execution_log = pd.concat([self.df_execution_log, df_daily_log], ignore_index=True)

                current_units = new_units
            else:
                days_until = self.rebalance_period - days_since_reb
                print(f"   [-] It is not rebalancing day. Using Buy & Hold method (there are {days_until} days until rebalancing).")
                current_leverage = leverage_yesterday

            etf_value = (current_units * cp).sum()
            new_nav = etf_value + current_cash
            new_weights = (current_units * cp) / new_nav

            daily_vt_return = np.log(new_nav / last_row['NAV_close'])
            past_vt_returns = self.state_ledger['ep_ptf_ret'].tail(self.vol_lookback -1).copy()
            combined_vt_returns = pd.concat([past_vt_returns, pd.Series([daily_vt_return])])
            current_vt_vol = combined_vt_returns.std()

            last_close_prices = pd.Series([last_row[f"Close_{tk}"] for tk in tickers], index=tickers)
            bod_values = current_units * last_close_prices
            bod_nav = bod_values.sum() + last_row.get("Cash", 0.0)
            bod_weights = (bod_values / bod_nav).fillna(0)
            window_size = 21
            mu_vector = self.df_returns.iloc[-window_size:].mean()
            ea_ptf_ret = (bod_weights * mu_vector).sum()
                      
            
        elif self.method == "calendar_rebalancing":
            base_weights = self.prf.loc[tickers]
            current_units = pd.Series([last_row[f"Units_{tk}"] for tk in tickers], index=tickers)
            current_cash = last_row.get("Cash", 0.0)
            bps = getattr(self, 'bps', 0.0005)
            
            units_cols = [f"Units_{tk}" for tk in tickers]
            df_units_history = self.state_ledger[units_cols]
            change = df_units_history.ne(df_units_history.shift(1)).any(axis=1)
            reb_days = change[change].index
            
            if len(reb_days) > 0:
                last_exec_day = reb_days[-1]
                days_since_exec = len(self.state_ledger.loc[last_exec_day:])
            else:
                days_since_exec = self.rebalance_period 

            if days_since_exec >= self.rebalance_period:
                print(f"   [!] Calendar rebalancing ({days_since_exec} days since last rebalancing).")
                
                yesterday_nav = last_row['NAV_close']
                target_allocations = yesterday_nav * base_weights

                # --- ESTIMATION LEDGER (Hypothetical execution at Yesterday's Close) ---
                yesterday_close = pd.Series([last_row[f"Close_{tk}"] for tk in tickers], index=tickers)
                est_new_units = target_allocations / yesterday_close
                est_units_trade = est_new_units - current_units
                
                is_est_sell = est_units_trade < -1e-6
                is_est_buy = est_units_trade > 1e-6

                est_cash_sell = (abs(est_units_trade[is_est_sell]) * yesterday_close[is_est_sell]).sum()
                est_cash_buy = (abs(est_units_trade[is_est_buy]) * yesterday_close[is_est_buy]).sum()

                est_trade_values = abs(est_units_trade) * yesterday_close
                est_fee_per_asset = est_trade_values * bps
                est_trade_fees = est_fee_per_asset.sum()

                est_available_cash = current_cash + est_cash_sell
                est_cash_required = est_cash_buy + est_trade_fees
                
                est_injected_capital = 0.0
                if est_available_cash >= est_cash_required:
                    est_cash_balance_end = est_available_cash - est_cash_required
                else:
                    est_shortfall = est_cash_required - est_available_cash
                    est_injected_capital = est_shortfall
                    est_cash_balance_end = -est_shortfall

                if getattr(self, 'df_estimation_log', None) is None:
                    self.df_estimation_log = pd.DataFrame()
                    
                daily_est_log = []
                for asset in base_weights.index:
                    action = 'Hold'
                    if is_est_buy[asset]: action = 'Buy'
                    elif is_est_sell[asset]: action = 'Sell'
                    
                    daily_est_log.append({
                        'Decision_Date': self.state_ledger.index[-1],
                        'NAV_close': yesterday_nav,
                        'Asset': asset,
                        'Target Weight': base_weights[asset],
                        'Estimated Action': action,
                        'Estimated Trade Units': est_units_trade[asset],
                        'Estimated Fees': est_fee_per_asset[asset],
                        'Est_Cash_Balance': est_cash_balance_end,
                        'Est_Capital_Injection': est_injected_capital
                    })
                
                if daily_est_log:
                    df_daily_est = pd.DataFrame(daily_est_log)
                    self.df_estimation_log = pd.concat([self.df_estimation_log, df_daily_est], ignore_index=True)

                # --- EXECUTION --- 
                new_units = target_allocations / op
                units_trade = new_units - current_units
                
                is_sell = units_trade < -1e-6
                is_buy = units_trade > 1e-6

                cash_sell = (abs(units_trade[is_sell]) * op[is_sell]).sum()
                cash_buy = (abs(units_trade[is_buy]) * op[is_buy]).sum()

                trade_values = abs(units_trade) * op
                fee_per_asset = trade_values * self.bps
                trade_fees = fee_per_asset.sum()

                available_cash = current_cash + cash_sell
                cash_required = cash_buy + trade_fees
                
                injected_capital = 0.0
                
                if available_cash >= cash_required:
                    current_cash = available_cash - cash_required
                else:
                    shortfall = cash_required - available_cash
                    injected_capital = shortfall
                    current_cash = -shortfall

                post_trade_etf_values = new_units * op
                post_trade_nav = post_trade_etf_values.sum() + current_cash
                post_trade_weights = post_trade_etf_values / post_trade_nav
                
                if getattr(self, 'df_execution_log', None) is None:
                    self.df_execution_log = pd.DataFrame()
                    
                daily_log = []
                for asset in base_weights.index:
                    trade_qty = units_trade[asset]
                    action = 'Hold'
                    if is_buy[asset]: 
                        action = 'Buy'
                    elif is_sell[asset]: 
                        action = 'Sell'
                    
                    daily_log.append({           
                        'Decision_Date': self.state_ledger.index[-1],
                        'Execution_Date': new_date,
                        'Nav_post_trade' : post_trade_nav,
                        'Asset': asset,
                        'Target Weight': base_weights[asset],
                        'True Weight': post_trade_weights[asset],
                        'Action': action,
                        'Trade Units': trade_qty,
                        'Fees': fee_per_asset[asset],
                        'Cash_Balance': current_cash,
                        'Capital_Injected': injected_capital,
                        'Units_After': new_units[asset]      
                    })
                
                if daily_log:
                    df_daily_log = pd.DataFrame(daily_log)
                    self.df_execution_log = pd.concat([self.df_execution_log, df_daily_log], ignore_index=True)

                current_units = new_units
            else:
                days_until = self.rebalance_period - days_since_exec
                print(f"   [-] It is not rebalancing day. Buy and hold (For {days_until} days).")
                
            etf_value = (current_units * cp).sum()
            new_nav = etf_value + current_cash
            new_weights = (current_units * cp) / new_nav  

            yesterday_nav = last_row["NAV_close"]
            ep_ptf_ret = (new_nav / yesterday_nav) - 1.0 if yesterday_nav > 0 else 0.0

            last_close_prices = pd.Series([last_row[f"Close_{tk}"] for tk in tickers], index=tickers)
            bod_values = current_units * last_close_prices
            bod_nav = bod_values.sum() + last_row.get("Cash", 0.0)
            bod_weights = (bod_values / bod_nav).fillna(0)
            window_size = 21
            mu_vector = self.df_returns.iloc[-window_size:].mean()
            ea_ptf_ret = (bod_weights * mu_vector).sum()          


        elif self.method == "drift_rebalancing":
            base_weights = self.prf.loc[tickers]
            current_units = pd.Series([last_row[f"Units_{tk}"] for tk in tickers], index=tickers)
            current_cash = last_row.get("Cash", 0.0)
            bps = getattr(self, 'bps', 0.0005)
            
            if not hasattr(self, 'drift_band'):
                raise AttributeError("Drift is not defined")

            yesterday_weights = pd.Series([last_row[f"Weight_{tk}"] for tk in tickers], index=tickers)
            yesterday_nav = last_row['NAV_close']
            
            drift = yesterday_weights - base_weights
            
            # TRIGGER
            if (drift.abs() > self.drift_band).any():
                print(f"   [!] Tolerance band surpassed in the last close. Drift rebalancing should be used.")
                
                target_allocations = yesterday_nav * base_weights

                # --- ESTIMATION LEDGER (Hypothetical execution at Yesterday's Close) ---
                yesterday_close = pd.Series([last_row[f"Close_{tk}"] for tk in tickers], index=tickers)
                est_new_units = target_allocations / yesterday_close
                est_units_trade = est_new_units - current_units
                
                is_est_sell = est_units_trade < -1e-6
                is_est_buy = est_units_trade > 1e-6

                est_cash_sell = (abs(est_units_trade[is_est_sell]) * yesterday_close[is_est_sell]).sum()
                est_cash_buy = (abs(est_units_trade[is_est_buy]) * yesterday_close[is_est_buy]).sum()

                est_trade_values = abs(est_units_trade) * yesterday_close
                est_fee_per_asset = est_trade_values * bps
                est_trade_fees = est_fee_per_asset.sum()

                est_available_cash = current_cash + est_cash_sell
                est_cash_required = est_cash_buy + est_trade_fees
                
                est_injected_capital = 0.0
                if est_available_cash >= est_cash_required:
                    est_cash_balance_end = est_available_cash - est_cash_required
                else:
                    est_shortfall = est_cash_required - est_available_cash
                    est_injected_capital = est_shortfall
                    est_cash_balance_end = -est_shortfall

                if getattr(self, 'df_estimation_log', None) is None:
                    self.df_estimation_log = pd.DataFrame()
                    
                daily_est_log = []
                for asset in base_weights.index:
                    action = 'Hold'
                    if is_est_buy[asset]: action = 'Buy'
                    elif is_est_sell[asset]: action = 'Sell'
                    
                    daily_est_log.append({
                        'Decision_Date': self.state_ledger.index[-1],
                        'NAV_close': yesterday_nav,
                        'Asset': asset,
                        'Target Weight': base_weights[asset],
                        'Estimated Action': action,
                        'Estimated Trade Units': est_units_trade[asset],
                        'Estimated Fees': est_fee_per_asset[asset],
                        'Est_Cash_Balance': est_cash_balance_end,
                        'Est_Capital_Injection': est_injected_capital
                    })
                
                if daily_est_log:
                    df_daily_est = pd.DataFrame(daily_est_log)
                    self.df_estimation_log = pd.concat([self.df_estimation_log, df_daily_est], ignore_index=True)
                
                # --- EXECUTION ---
                new_units = target_allocations / op
                units_trade = new_units - current_units
                
                is_sell = units_trade < -1e-6
                is_buy = units_trade > 1e-6

                cash_sell = (abs(units_trade[is_sell]) * op[is_sell]).sum()
                cash_buy = (abs(units_trade[is_buy]) * op[is_buy]).sum()

                trade_values = abs(units_trade) * op
                fee_per_asset = trade_values * bps
                trade_fees = fee_per_asset.sum()

                available_cash = current_cash + cash_sell
                cash_required = cash_buy + trade_fees
                
                injected_capital = 0.0
                
                if available_cash >= cash_required:
                    current_cash = available_cash - cash_required
                else:
                    shortfall = cash_required - available_cash
                    injected_capital = shortfall
                    current_cash = -shortfall
                
                post_trade_etf_values = new_units * op
                post_trade_nav = post_trade_etf_values.sum() + current_cash
                post_trade_weights = post_trade_etf_values / post_trade_nav
                    
                if getattr(self, 'df_execution_log', None) is None:
                    self.df_execution_log = pd.DataFrame()
                    
                daily_log = []
                for asset in base_weights.index:
                    trade_qty = units_trade[asset]
                    action = 'Hold'
                    if is_buy[asset]: 
                        action = 'Buy'
                    elif is_sell[asset]: 
                        action = 'Sell'
                    
                    daily_log.append({        
                        'Decision_Date': self.state_ledger.index[-1],
                        'Execution_Date': new_date,
                        'Nav_post_trade' : post_trade_nav,
                        'Asset': asset,
                        'Target Weight': base_weights[asset],
                        'True Weight': post_trade_weights[asset],
                        'Action': action,
                        'Trade Units': trade_qty,
                        'Fees': fee_per_asset[asset],
                        'Cash_Balance': current_cash,
                        'Capital_Injected': injected_capital,
                        'Units_After': new_units[asset]      
                    })
                
                if daily_log:
                    df_daily_log = pd.DataFrame(daily_log)
                    self.df_execution_log = pd.concat([self.df_execution_log, df_daily_log], ignore_index=True)

                current_units = new_units

            else:
                max_drift_pct = drift.abs().max() * 100
                print(f"   [-] Maximum deviation ({max_drift_pct:.2f}%) inside the band {self.drift_band*100}%. No alterations were made.")

            etf_value = (current_units * cp).sum()
            new_nav = etf_value + current_cash
            new_weights = (current_units * cp) / new_nav

            yesterday_nav = last_row["NAV_close"]
            ep_ptf_ret = (new_nav / yesterday_nav) - 1.0 if yesterday_nav > 0 else 0.0

            last_close_prices = pd.Series([last_row[f"Close_{tk}"] for tk in tickers], index=tickers)
            bod_values = current_units * last_close_prices
            bod_nav = bod_values.sum() + last_row.get("Cash", 0.0)
            bod_weights = (bod_values / bod_nav).fillna(0)
            window_size = 21
            mu_vector = self.df_returns.iloc[-window_size:].mean()
            ea_ptf_ret = (bod_weights * mu_vector).sum()
        

        # ==========================================
        # ADD NEW LINE
        # ==========================================

        new_record = {
            'NAV_close': new_nav,
            'ep_ptf_ret': ep_ptf_ret,
            'ea_ptf_ret': ea_ptf_ret,
            'Cash': current_cash
        }

        if self.method == "volatility_targeting":
            new_record['Ref_Market_Vol'] = current_ref_vol
            new_record['Actual_VT_Vol'] = current_vt_vol
            new_record['Target_Vol'] = current_target
            new_record['Leverage'] = current_leverage
        
        for tk in tickers:
            new_record[f'Open_{tk}'] = op[tk]
            new_record[f'Close_{tk}'] = cp[tk]
            new_record[f'Units_{tk}'] = current_units[tk]  
            new_record[f'Weight_{tk}'] = new_weights[tk]

        df_new_row = pd.DataFrame([new_record], index=[new_date])
        df_new_row.index.name = 'Data'
        
        for col in self.state_ledger.columns:
            if col not in df_new_row.columns:
                df_new_row[col] = np.nan
        
        self.state_ledger = pd.concat([self.state_ledger, df_new_row], axis=0)
        self.is_dirty = True




    # =========================================================================
    # MEMORY AND EXPORTATION
    # =========================================================================


    def load_from_xlsx(self, filename):
        """ Verifies the existence of a previous portfolio """
        if os.path.exists(filename):
            print(f"\n [MEMORY] '{filename}' found. ")
            self.state_ledger = pd.read_excel(filename, sheet_name="Visao_Geral", index_col="Data", parse_dates=True)
            return True
        else:
            print(f"\n[MEMORY] No file '{filename}' found. Builidng a new...")
            return False



    def save_to_xlsx(self, filename):
        """ Keeps the file in an excel, if there is new data to keep"""

        if not self.is_dirty:
            print("\n-> No new data. No file was created.")
            return
        
        if self.state_ledger is not None:
            try:
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    
                    # 1. Guardar a Visão Geral (State Ledger)
                    df_export = self.state_ledger.copy()
                    df_export = df_export.round(3)
                    df_export.index = df_export.index.strftime('%Y-%m-%d')
                    df_export.to_excel(writer, sheet_name="Visao_Geral")
                    
                    # 2. Guardar o Execution Log
                    if getattr(self, 'df_execution_log', None) is not None and not self.df_execution_log.empty:
                        df_log_export = self.df_execution_log.copy()
                        
                        for col in ['Decision_Date', 'Execution_Date']:
                            if col in df_log_export.columns:
                                df_log_export[col] = pd.to_datetime(df_log_export[col]).dt.strftime('%Y-%m-%d')
                        
                        cols_to_round = ['NAV_close', 'Nav_post_trade', 'Trade Units', 'Fees', 'Cash_Balance', 'Capital_Injected', 'Units_After']
                        for col in cols_to_round:
                            if col in df_log_export.columns:
                                df_log_export[col] = df_log_export[col].astype(float).round(3)

                        df_log_export.to_excel(writer, sheet_name="Execution_Log", index=False)
                            
                    # 3. Guardar o Estimation Log
                    if getattr(self, 'df_estimation_log', None) is not None and not self.df_estimation_log.empty:
                        df_est_export = self.df_estimation_log.copy()
                        
                        if 'Decision_Date' in df_est_export.columns:
                            df_est_export['Decision_Date'] = pd.to_datetime(df_est_export['Decision_Date']).dt.strftime('%Y-%m-%d')
                        
                        cols_to_round = ['NAV_close', 'Target Weight', 'Estimated Trade Units', 'Estimated Fees', 'Est_Cash_Balance', 'Est_Capital_Injection']
                        for col in cols_to_round:
                            if col in df_est_export.columns:
                                df_est_export[col] = df_est_export[col].astype(float).round(3)

                        df_est_export.to_excel(writer, sheet_name="Estimation_Log", index=False)
                        
                print(f"[SUCESS] Portfolio and logs kept on {filename}\n")
                
            except PermissionError:
                print(f"\n[ERROR] '{filename}' is open in Excel!")
                print("Please, close excel and run again")
            except ModuleNotFoundError:
                print("\n[ERROR] 'openpyxl' is not installed. Execute in the terminal: pip install openpyxl")