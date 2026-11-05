import numpy as np
import pandas as pd
import yfinance as yf
import os
import matplotlib.pyplot as plt

from dataprep import Data
from portfolio import Portfolio
from ex_ante import ExAnte
from ex_post import ExPost
from etf import ETFScreener
from monte_carlo import MonteCarlo


def main():
    # =========================================================================
    # SETTINGS
    # =========================================================================
    excel_file = "file2.xlsx"

    # choose the name of the output file
    output_file = "Dn_cr_vol21_reb21.xlsx" # portfolio constructed using your etfs
    bm_output_file = "Dn_cr_vol21_reb21_bm.xlsx" # portfolio constructed using their benchamrks

    # don't change
    etf_col = "etf"
    bm_col = "track"
    class_col = "class"
    rf_col = "risk free"
    
    # introduce the desired initial capital
    initial_capital = 10000

    # introduce the desired profile
    selected_profile = "your_profile"
    
    # choose the deisred method of construction: "buy_hold" , "naive_risk_parity" , "volatility_targeting" , "calendar_rebalancing", "drift_rebalancing"
    construction_method = "calendar_rebalancing"

    # If the portfolio is being created for the first time:
    # True = download historical data and build from there
    # False = build from today's date only
    use_historic = True

    # If True, choose the start and end dates for the backtest
    start_date = "2020-01-01" 
    end_date = "2026-01-01"

    # Monte Carlo: estimation uses dates <= cutoff, realized comparison uses dates > cutoff
    mc_cutoff = pd.Timestamp("2024-12-31")

    # Do you want to append today's data to the portfolio?
    # True = Updates the table with the latest market close.
    # False = Only generates/maintains the static historical backtest.
    add_today = False
    
    # Parameter Dictionary: Add any variables for future strategies here
    strategy_params = {
        "vol_lookback": 21,
        "rebalance_period": 21,
        "bps" : 0.0005,
        "drift_band" : 0.05,
        "confidence_level" : 0.95,
        "time_horizon" : 252,
        "start_date": start_date,
        "train_end": end_date,
        "init_cap": initial_capital
    }
    

    # Compute Module x measures?
    mod1 = False
    mod2 = True
    ex_ante = False
    ex_post = False
    monte_carlo_analysis = True




    # =========================================================================
    # PORTFOLIO CONSTRUCTION
    # =========================================================================

    data = Data(file=excel_file, prf = selected_profile, etf_col=etf_col, bm_col=bm_col, class_col=class_col, rf_col=rf_col, start_date=start_date, end_date=end_date)
    
    data.profile_df()
    data.extract_data()
    data.risk_free()
    data.asset_returns()

    etfs = data.etf_list
    bms = data.bm_list



    # Verification of portfolio existence
    historic_exists = os.path.exists(output_file)

    if historic_exists:
        print(f"\n-> File '{output_file}' found. Loading memory...")

        empty_df = pd.DataFrame(columns=etfs)
        
        # Object creation with parameter unpacking (**strategy_params)
        ptf = Portfolio(
            df_profile=data.df_profile, df_bm_close = empty_df, df_close=empty_df, df_open=empty_df, df_returns=empty_df, df_rf=data.df_rf,
            profile=selected_profile, method=construction_method, V0=initial_capital,
            **strategy_params
        )
        ptf.load_from_xlsx(output_file)


        if add_today:
            print("   Updating data with today's market close...")
            data_today = yf.download(etfs, period="1d", progress=False, auto_adjust=True)
            
            prices_today = data_today["Close"][etfs].iloc[-1]
            open_today = data_today["Open"][etfs].iloc[-1]
            date_today = prices_today.name
            
            ptf.append_daily_metrics(date=date_today, cp=prices_today, op=open_today)

        else:
            print("   [Backtest Mode] Daily update inactive.")

    else:
        print(f"\n-> Excel does not exist. Creating portfolio...")
        
        if use_historic:
            print(f"   [Historic Mode] Simulating the past from {start_date} to {end_date}...")
            #data.extract_data()
            data.asset_returns() 
            
            df_c = data.df_close
            df_bm_c= data.df_bm_close
            df_o = data.df_open
            df_r = data.df_returns
            
        else:
            print("   [Start Today Mode] No history. Initializing with current data...")
            data_today = yf.download(etfs, period="1d", progress=False, auto_adjust=True)
            bm_data_today = yf.download(bms, period="1d", progress=False, auto_adjust=True)
            
            cl = data_today["Close"][etfs]
            bm_cl=bm_data_today["Close"][bms]
            op = data_today["Open"][etfs]

            if isinstance(cl, pd.Series):
                cl = cl.to_frame(name=etfs[0])
                bm_cl = bm_cl.to_frame(name=bms[0])
                op = op.to_frame(name=etfs[0])

            df_c = cl.tail(1)
            df_bm_c = bm_cl.tail(1)
            df_o = op.tail(1)
            df_r = pd.DataFrame(columns=etfs) # No history, returns start empty


        # Portfolio construction with parameter unpacking (**strategy_params)
        ptf = Portfolio(
            df_profile=data.df_profile, df_bm_close=df_bm_c, df_close=df_c, df_open=df_o, df_returns=df_r,df_rf=data.df_rf,
            profile=selected_profile, method=construction_method, V0=initial_capital,
            **strategy_params
        )
        ptf.build_portfolio()

    

        # If it is historic AND we want to append today:
        if use_historic and add_today:
            print("   Appending today's data to the history...")
            data_today = yf.download(etfs, period="1d", progress=False, auto_adjust=True)
            
            prices_today = data_today["Close"][etfs].iloc[-1]
            open_today = data_today["Open"][etfs].iloc[-1]
            date_today = prices_today.name
            
            ptf.append_daily_metrics(date=date_today, cp=prices_today, op=open_today)



    # Verification of portfolio existence
    bm_historic_exists = os.path.exists(bm_output_file)

    if bm_historic_exists:
        print(f"\n-> File '{bm_output_file}' found. Loading memory...")

        empty_df = pd.DataFrame(columns=bms)
        
        # Object creation with parameter unpacking (**strategy_params)
        # benchamrk portfolio
        # Object creation with parameter unpacking (**strategy_params)
        bm_ptf = Portfolio(
            df_profile=data.df_bm_profile, df_bm_close=empty_df, df_close=empty_df, df_open=empty_df, df_returns=empty_df, df_rf=data.df_rf,
            profile=selected_profile, method=construction_method, V0=initial_capital,
            **strategy_params
        )
        bm_ptf.load_from_xlsx(bm_output_file)

        if add_today:
            print("   Updating data with today's market close...")

            # benchmark portfolio
            bm_data_today = yf.download(bms, period="1d", progress=False, auto_adjust=True)
            
            bm_prices_today = bm_data_today["Close"][bms].iloc[-1]
            bm_open_today = bm_data_today["Open"][bms].iloc[-1]
            bm_date_today = bm_prices_today.name
            
            
            bm_ptf.append_daily_metrics(date=bm_date_today, cp=bm_prices_today, op=bm_open_today)
        else:
            print("   [Backtest Mode] Daily update inactive.")

    else:
        print(f"\n-> Excel does not exist. Creating portfolio...")
        
        if use_historic:
            print(f"   [Historic Mode] Simulating the past from {start_date} to {end_date}...")
            #data.extract_data()
            data.asset_returns() 
            

            df_bm_c = data.df_bm_close
            df_bm_o = data.df_bm_open
            df_bm_r = data.df_bm_returns
            
        else:
            print("   [Start Today Mode] No history. Initializing with current data...")
            bm_data_today = yf.download(bms, period="1d", progress=False, auto_adjust=True)

            bm_cl = bm_data_today["Close"][bms]
            bm_op = bm_data_today["Open"][bms]

            if isinstance(bm_cl, pd.Series):
                bm_cl =bm_cl.to_frame(name=bms[0])
                bm_op = bm_op.to_frame(name=bms[0])

            df_bm_c = bm_cl.tail(1)
            df_bm_o = bm_op.tail(1)
            df_bm_r = pd.DataFrame(columns=bms) # No history, returns start empty

        # Portfolio construction with parameter unpacking (**strategy_params)

        bm_ptf = Portfolio(
            df_profile=data.df_bm_profile, df_bm_close=df_bm_c, df_close=df_bm_c, df_open=df_bm_o, df_returns=df_bm_r, df_rf=data.df_rf,
            profile=selected_profile, method=construction_method, V0=initial_capital,
            **strategy_params
        )
        bm_ptf.build_portfolio()
    

        # If it is historic AND we want to append today:
        if use_historic and add_today:
            print("   Appending today's data to the history...")

            bm_data_today = yf.download(bms, period="1d", progress=False, auto_adjust=True)
            
            bm_prices_today = bm_data_today["Close"][bms].iloc[-1]
            bm_open_today = bm_data_today["Open"][bms].iloc[-1]
            bm_date_today = bm_prices_today.name

            bm_ptf.append_daily_metrics(date=bm_date_today, cp=bm_prices_today, op=bm_open_today)


    # =========================================================================
    # SAVE IN EXCEL
    # =========================================================================
    ptf.save_to_xlsx(filename=output_file)
    print("\n-> Process completed! The State Ledger is ready.")

    bm_ptf.save_to_xlsx(filename=bm_output_file)
    print("\n-> Process completed! The benchmark portfolio State Ledger is ready.")


    # =========================================================================
    # EX POST
    # =========================================================================
    
    filename5 = "D_bh.xlsx"
    if os.path.exists(filename5):
        print(f"\n[MEMORY] '{filename5}' found.")
        state_ledger5 = pd.read_excel(filename5, sheet_name="Visao_Geral", index_col="Data", parse_dates=True)
    else:
        print(f"\n[MEMORY] No file '{filename5}' found.")
        state_ledger5 = pd.DataFrame()

    ep_metrics = ExPost(df_profile=data.df_profile, profile=selected_profile, df_returns=data.df_returns, df_rf=data.df_rf, ptf_ret=ptf.state_ledger['ep_ptf_ret'], ptf_nav=ptf.state_ledger['NAV_close'], bm_ret=bm_ptf.state_ledger['ep_ptf_ret'], **strategy_params)
    ea_metrics = ExAnte(df_profile=data.df_profile, profile=selected_profile, ptf_state_ledger=ptf.state_ledger, bm_state_ledger = bm_ptf.state_ledger, df_returns=data.df_returns, df_rf=data.df_rf, ptf_ret=ptf.state_ledger['ea_ptf_ret'], ptf_nav=ptf.state_ledger['NAV_close'], bm_ret=bm_ptf.state_ledger['ea_ptf_ret'],V0=initial_capital, **strategy_params)

    

    # ============================
    # MODULE 1
    # ============================

    if mod1 == True:
        data.class_allocation()
        print(f"\n -> Class alocation ptf: {data.cl_alloc[selected_profile]}")

        ep_metrics.max_drawdown()
        mdd = round(ep_metrics.mdd,3)
        print(f"\n -> Maximum historical Drawdown:{ep_metrics.mdd:.3f}")

        ep_metrics.annualized_vol()
        av=round(ep_metrics.ptf_ann_vol,3)
        print(f"\n -> Annualized Vol (using all data available): {ep_metrics.ptf_ann_vol:.3f}")

        ep_metrics.rolling_annualized_vol()
        rav=round( ep_metrics.roll_ptf_ann_vol,3)
        print(f"\n -> Rolling annualized volatility (computed with a lookback period of {ptf.vol_lookback} days): {ep_metrics.roll_ptf_ann_vol.iloc[-1]:.3f}")
 
        mod_df = pd.DataFrame({
            'class alocation': data.cl_alloc[selected_profile],
            'Maximum Historical Drawdown': mdd,
            'Annualized Volatility': av,
            'Rolling Annualized Volatility' : rav
        })

        print(mod_df)
        

    if mod2 == True:
        print(f"Correlation Matrix")

        ep_metrics.correlation_matrix()
        print(f"\n Correlation heatmap: {ep_metrics.axis_corr}")
        plt.show()

        # ---- ETF screening / scoring pipeline (scd.py) ----
        # Requires a 'ter' column in file2.xlsx (ETFScreener default ter_col="ter").
        os.makedirs("excels_2", exist_ok=True)

        screener = ETFScreener(file2=excel_file, ticker_col=etf_col, track_col=bm_col,
                               class_col=class_col, rf_col=rf_col)
        screener.read()

        # 2. Prices (close, volume, benchmark close)
        screener.extract(start_date=start_date, end_date=end_date)

        # 3. Metrics
        df_metrics = screener.compute_metrics()
        print("\n=== Full ETF metrics (Table A material) ===")
        print(df_metrics.to_string(index=False))
        screener.save_metrics("excels_2/etf_metrics_full.xlsx")                     

        # 4. Eligibility
        df_eligible = screener.apply_eligibility()                 # default thresholds; pass kwargs to override
        print("\n=== Eligibility filtering ===")
        print(df_eligible[["etf", "track", "scoring_group", "eligible",
                   "non_eligibility_reason"]].to_string(index=False))
        
        # 5. Scoring
        df_score = screener.score()
        print("\n=== Final ETF scoring (Table A) ===")
        print(df_score.to_string(index=False))
        screener.save_scores("excels_2/score3.xlsx")

        # 6. Class allocation per profile (Table B)
        df_class_allocation = screener.class_allocation()
        print("\n=== Table B: target class allocation per profile ===")
        print(df_class_allocation.to_string(index=False))
        screener.save_class_allocation("excels_2/class_allocation.xlsx")

        print("\n=== Top-ranked ETF per scoring group ===")
        print(screener.top_picks.to_string(index=False))



    if ex_ante == True:
        
        print("[EX-ANTE METRICS]")

        ea_metrics.value_at_risk()
        print(f"\n -> Value at Risk: {ea_metrics.var:.3f} ")

        ea_metrics.conditional_value_at_risk()
        print(f"\n -> Conditional Value at Risk: {ea_metrics.cvar:.3f} ")

        ea_metrics.annualized_vol()
        print(f"\n -> Annualized Vol: {ea_metrics.ptf_ann_vol:.3f}")

        #plot do rolling annualized volatility

        ea_metrics.tracking_error()
        print(f"\n -> Tracking error in comparison with benchmark is: {ea_metrics.te:.3f}" )


    if ex_post == True:
        
        print("[EX POST Evaluation]")

        print(f"\n RISK")
        ep_metrics.max_drawdown()
        print(f"\n -> Maximum historical Drawdown:{ep_metrics.mdd:.3f}")

        ep_metrics.annualized_vol()
        print(f"\n -> Annualized Vol (using all data available): {ep_metrics.ptf_ann_vol:.3f}")

        ep_metrics.rolling_annualized_vol()
        print(f"\n -> Rolling annualized volatility (computed with a lookback period of {ptf.vol_lookback} days): {ep_metrics.roll_ptf_ann_vol.iloc[-1]:.3f}")

        #plot da rolling_annualized_vol

        print(f"\n PERFORMANCE")

        ep_metrics.excess_returns_rf()
        print(f"\n -> ETFs portfolio mean excess returns: {ep_metrics.er_rf.mean():.5f}")
        print(f"\n -> BMs portfolio mean excess returns: {ep_metrics.er_rf_bm.mean():.5f}")

        ep_metrics.beta_alpha()
        print(f"\n -> Alpha: {ep_metrics.alpha:.3f}")
        print(f"\n -> Beta: {ep_metrics.beta:.3f}")

        ep_metrics.tracking_error()
        print(f"\n -> Ex Post Annualized tracking error - strategy vs benchmark: {ep_metrics.te_ann:.3f}")


    if monte_carlo_analysis == True:
        # ---- Monte Carlo out-of-sample backtest (monte_carlo_last.py) ----
        # mu and cov estimated ONLY from returns <= mc_cutoff; simulation horizon
        # sized to the realized window > mc_cutoff; realized buy and hold NAV
        # overlaid on the simulated fan. Reuses the 'data' object built above.
        print("\n[MONTE CARLO - out-of-sample backtest]")
        os.makedirs("monte_carlo", exist_ok=True)

        est_ret = data.df_returns[data.df_returns.index <= mc_cutoff]
        realized_close = data.df_close[data.df_close.index > mc_cutoff]
        horizon = len(realized_close)
        if horizon == 0:
            raise ValueError("No realized data after the cutoff to compare against.")

        mu = est_ret.mean()
        cov = est_ret.cov()
        print(f"Estimation window: {len(est_ret)} days (<= {mc_cutoff.date()})")
        print(f"Realized horizon : {horizon} days (> {mc_cutoff.date()})")

        # NOTE: MonteCarlo expects 'time horizon' / 'number of paths' (with spaces),
        # NOT strategy_params keys, so do not unpack **strategy_params here.
        mc = MonteCarlo(df_profile=data.df_profile, profile=selected_profile,
                        df_returns=est_ret, mean=mu, var=None, cov=cov,
                        V0=initial_capital, seed=42,
                        **{"time horizon": horizon, "number of paths": 5000,
                           "cutoff_date": mc_cutoff})
        mc._prepare_params()   # establishes mc.assets / mc.weights order

        # Realized buy and hold NAV, aligned to mc.assets, starting at V0 on the cutoff
        base_price = data.df_close[data.df_close.index <= mc_cutoff].iloc[-1].reindex(mc.assets)
        g_real = realized_close[mc.assets].divide(base_price)
        nav_real = initial_capital * (g_real.to_numpy() @ mc.weights)
        nav_real = np.concatenate([[initial_capital], nav_real])   # day 0 = cutoff = V0

        mc.simulate()
        summary = mc.analyse()

        sim_terminal = mc.ptf_nav[-1]
        term_pctile = (sim_terminal < nav_real[-1]).mean() * 100.0
        real_mdd = abs(mc._drawdown_stats(nav_real.reshape(-1, 1))["mdd"][0])
        mdd_p5, mdd_p95 = np.percentile(np.abs(mc.ptf_mdd), [5, 95])

        print("\n=== Portfolio vs asset MDD summary (simulated) ===")
        print(summary)
        print("\n=== Out-of-sample check (realized vs simulated) ===")
        print(f"realized terminal NAV: {nav_real[-1]:.0f}  "
              f"(simulated median {np.median(sim_terminal):.0f}), "
              f"landed at simulated percentile {term_pctile:.1f}")
        print(f"realized MDD: {real_mdd:.2%}  simulated p5 to p95: [{mdd_p5:.2%}, {mdd_p95:.2%}]")
        print(f"realized MDD inside simulated p5 to p95: {mdd_p5 <= real_mdd <= mdd_p95}")

        summary.to_excel("monte_carlo/mc_portfolio_summary.xlsx", index=False)
        mc.plot_fan(realized_nav=nav_real,
                    realized_dates=[mc_cutoff] + list(realized_close.index),
                    fname="monte_carlo/mc_fan_vs_realized.png")
        mc.plot_portfolio(prefix="monte_carlo/mc_portfolio")
        mc.plot_assets("monte_carlo/mc_portfolio_vs_assets.png")
        print("\nSaved Monte Carlo outputs to monte_carlo/ folder")

if __name__ == "__main__":
    main()
