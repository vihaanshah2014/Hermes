import json
import os
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###################
# 1) JSON loading
###################
def load_json_file(filepath):
    """
    Load JSON from a file and return parsed content.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def fetch_all_symbols_from_jsons(djia_json_path, nasdaq_json_path):
    """
    Load the DJIA and NASDAQ JSON files, combine all symbols into one list.
    Return a list of (symbol, name) tuples, with duplicates removed by symbol.
    """
    djia_data = load_json_file(djia_json_path)
    nasdaq_data = load_json_file(nasdaq_json_path)

    djia_symbols = [(corp["symbol"], corp["name"]) for corp in djia_data["corporations"]]
    nasdaq_symbols = [(corp["symbol"], corp["name"]) for corp in nasdaq_data["corporations"]]

    # Combine & remove duplicates by symbol
    combined = {}
    for sym, nam in djia_symbols + nasdaq_symbols:
        combined[sym] = nam  # overwrites duplicates with last occurrence

    # Return list of (symbol, name)
    return list(combined.items())

###################
# 2) Bayesian forced-close logic
###################
def bayesian_update(prior_mean, prior_variance, observed_price, predicted_price, likelihood_variance):
    error = observed_price - predicted_price
    posterior_variance = 1.0 / (1.0 / prior_variance + 1.0 / likelihood_variance)
    posterior_variance = np.clip(posterior_variance, 1e-7, 1.0)

    posterior_mean = posterior_variance * (
        (prior_mean / prior_variance) + (error / likelihood_variance)
    )
    posterior_mean = np.clip(posterior_mean, -0.5, 0.5)
    return posterior_mean, posterior_variance

def predict_next_price(current_price, prior_mean):
    return current_price * (1.0 + prior_mean)

def trading_simulation_close_positions(prices_df, prior_mean, prior_variance, likelihood_variance, initial_capital):
    """
    Forced-close, day-by-day. Return final portfolio value or None if bust.
    """
    daily_prices = prices_df["Close"].values
    capital = initial_capital
    shares = 0.0
    short_shares = 0.0
    short_proceeds = 0.0

    for i in range(1, len(daily_prices)):
        current_price = daily_prices[i - 1]
        observed_price = daily_prices[i]

        # Predict
        predicted_price = predict_next_price(current_price, prior_mean)

        # Close existing position
        if shares > 0:
            capital += shares * observed_price
            shares = 0.0
        if short_shares > 0:
            cost_to_close = short_shares * observed_price
            capital -= cost_to_close
            short_shares = 0.0
            short_proceeds = 0.0

        # Open new position
        if predicted_price > current_price:
            shares_to_buy = capital / current_price
            shares = shares_to_buy
            capital -= shares_to_buy * current_price
        elif predicted_price < current_price:
            shares_to_short = capital / current_price
            short_shares = shares_to_short
            short_proceeds = shares_to_short * current_price
            capital += short_proceeds
        else:
            pass

        # Check bust
        long_value = shares * observed_price
        short_value = 0.0
        if short_shares > 0:
            cost_to_buy_back = short_shares * observed_price
            short_value = short_proceeds - cost_to_buy_back

        portfolio_value = capital + long_value + short_value
        if portfolio_value <= 0:
            return None  # BUST

        # Bayesian update
        prior_mean, prior_variance = bayesian_update(
            prior_mean, prior_variance, observed_price, predicted_price, likelihood_variance
        )

    # Final evaluation
    last_price = daily_prices[-1]
    final_long_value = shares * last_price
    final_short_value = 0.0
    if short_shares > 0:
        cost_to_buy_back = short_shares * last_price
        final_short_value = short_proceeds - cost_to_buy_back

    return capital + final_long_value + final_short_value

###################
# 3) Worker function (no concurrency)
###################
def process_symbol(sym, start_date, end_date):
    """
    Return a dict with:
      "symbol", "status", "start_date", "end_date",
      "buyhold_value", "forced_value"
    """
    out = {
        "symbol": sym,
        "status": None,
        "start_date": None,
        "end_date": None,
        "buyhold_value": None,
        "forced_value": None,
    }
    try:
        df = yf.download(sym, start=start_date, end=end_date, progress=False)
        if df.empty or "Close" not in df.columns:
            out["status"] = "NO_DATA"
            return out

        df = df[["Close"]].dropna().sort_index()
        if len(df) < 2:
            out["status"] = "NO_DATA"
            return out

        out["start_date"] = df.index[0].strftime("%Y-%m-%d")
        out["end_date"]   = df.index[-1].strftime("%Y-%m-%d")

        # Buy&Hold
        start_price = df["Close"].iloc[0]
        end_price   = df["Close"].iloc[-1]
        out["buyhold_value"] = (end_price / start_price) * 100.0

        # Forced close
        fc_val = trading_simulation_close_positions(
            df,
            prior_mean=0.0,
            prior_variance=0.01,
            likelihood_variance=0.005,
            initial_capital=100.0
        )
        if fc_val is None:
            out["status"] = "BUST"
        else:
            out["status"] = "OK"
            out["forced_value"] = fc_val

    except Exception:
        out["status"] = "ERROR"
    return out

###################
# 4) Main
###################
def main():
    # 4A) Gather symbols
    djia_path = "djia.json"
    nasdaq_path = "nasdaq.json"

    combined_list = fetch_all_symbols_from_jsons(djia_path, nasdaq_path)
    symbols = [x[0] for x in combined_list]
    total_syms = len(symbols)

    print(f"Found {total_syms} symbols from JSON.\n")

    # 4B) Ten-year range
    start_date = "2014-01-01"
    end_date   = "2024-12-26"
    print(f"Fetching data from {start_date} to {end_date}. This might skip future data if not available.\n")

    # 4C) Process each symbol in a loop
    results = []
    for i, sym in enumerate(symbols, start=1):
        print(f"Progress: {i}/{total_syms}  (Symbol: {sym})", end='\r')
        res_dict = process_symbol(sym, start_date, end_date)
        results.append(res_dict)

    print("\nAll symbols processed.\n")

    # 4D) Write "output.txt" with per-symbol info
    with open("output.txt", "w", encoding="utf-8") as out:
        out.write("--- Final Results (Buy&Hold vs Forced-Close) ---\n\n")

        # Track summary counts
        count_ok = 0
        count_bust = 0
        count_nodata = 0
        count_error = 0

        for r in results:
            sym = r["symbol"]
            s   = r["status"]
            st  = r["start_date"]
            en  = r["end_date"]
            bhv = r["buyhold_value"]
            fcv = r["forced_value"]

            if s == "OK":
                count_ok += 1
            elif s == "BUST":
                count_bust += 1
            elif s == "NO_DATA":
                count_nodata += 1
            else:
                count_error += 1

            # Format buy&hold
            if bhv is not None:
                bh_pct = bhv - 100.0
                bh_sign = "+" if bh_pct >= 0 else ""
                bh_str = f"{bhv:6.2f} ({bh_sign}{bh_pct:5.2f}%)"
            else:
                bh_str = "None"

            # Format forced
            if fcv is not None:
                fc_pct = fcv - 100.0
                fc_sign = "+" if fc_pct >= 0 else ""
                fc_str = f"{fcv:6.2f} ({fc_sign}{fc_pct:5.2f}%)"
            else:
                fc_str = "None"

            line = (
                f"Symbol={sym}, Status={s}, Start={st}, End={en}, "
                f"BuyHold={bh_str}, ForcedClose={fc_str}\n"
            )
            out.write(line)

        # Summary line
        out.write(f"\nAnalyzed symbols: {len(results)}\n")
        out.write(f"   OK: {count_ok}\n")
        out.write(f"   BUST: {count_bust}\n")
        out.write(f"   NO_DATA: {count_nodata}\n")
        out.write(f"   ERROR: {count_error}\n")

    # 4E) Minimal console summary & bar chart
    #     We'll gather "OK" symbols, compute the "excess" = (forced - buyhold).
    ok_symbols = []
    for r in results:
        if r["status"] == "OK":
            sym = r["symbol"]
            bh_val = r["buyhold_value"]   # final capital from B&H
            fc_val = r["forced_value"]    # final capital from forced-close
            bh_ret = bh_val - 100.0
            fc_ret = fc_val - 100.0
            excess = fc_ret - bh_ret
            ok_symbols.append((sym, bh_ret, fc_ret, excess))

    made_money_forced = sum(1 for (sym, bh, fc, ex) in ok_symbols if fc > 0)
    lost_money_forced = len(ok_symbols) - made_money_forced

    print("===== SUMMARY FOR 2014-01-01 to 2024-12-26 (OK symbols) =====")
    print(f"OK symbols: {len(ok_symbols)}")
    print(f"Forced-Close made money: {made_money_forced}, lost money: {lost_money_forced}")

    # Sort by "excess" descending
    ok_symbols_sorted = sorted(ok_symbols, key=lambda x: x[3], reverse=True)

    # If we have enough, let's make a bar chart of top 10 "excess" and bottom 10
    top_n = 10
    winners = ok_symbols_sorted[:top_n]  # biggest positive differences
    losers  = ok_symbols_sorted[-top_n:] if len(ok_symbols_sorted) > top_n else []

    # We'll create a single figure with two subplots: winners and losers
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Forced-Close vs. Buy&Hold: Top & Bottom 10 Excess Returns (%)")

    # Bar chart for winners
    # x axis = symbol, y axis = 'excess'
    winners_syms   = [x[0] for x in winners]
    winners_excess = [x[3] for x in winners]
    ax1.barh(winners_syms[::-1], winners_excess[::-1], color='green')  # reversed so biggest on top
    ax1.set_title("Top 10 Excess (Forced - B&H)")
    ax1.set_xlabel("Excess Return (%)")
    ax1.set_ylabel("Symbol")

    # Bar chart for losers
    losers_syms   = [x[0] for x in losers]
    losers_excess = [x[3] for x in losers]
    ax2.barh(losers_syms[::-1], losers_excess[::-1], color='red')
    ax2.set_title("Bottom 10 Excess (Forced - B&H)")
    ax2.set_xlabel("Excess Return (%)")

    plt.tight_layout()
    plt.show()

    print("\nDone! Check 'output.txt' for all details, and see the bar chart for top/bottom excess.")


if __name__ == "__main__":
    main()
