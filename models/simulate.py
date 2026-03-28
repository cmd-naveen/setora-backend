"""
Tennis Prediction — Phase 3: Simulation Engine (Backtester)
============================================================
Walk-forward simulation of betting strategies using the trained model.
Tests realistic bankroll management, tracks P&L, drawdowns, and ROI.

Strategies:
  1. Flat stake (fixed % of initial bankroll)
  2. Kelly criterion (optimal growth)
  3. Fractional Kelly (1/4 Kelly — safer)

Usage:
  cd models/
  python simulate.py
"""

import json
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("simulate")

# ============================================================
# PATHS & CONFIG
# ============================================================
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
MODEL_PATH = OUTPUT_DIR / "tennis_model.pkl"
FEATURES_PATH = OUTPUT_DIR / "features.csv"
SIM_OUTPUT_DIR = OUTPUT_DIR / "simulation"
SIM_OUTPUT_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "elo_diff", "elo_surf_diff", "p1_elo", "p2_elo",
    "rank_diff", "log_rank_ratio", "rank_pts_diff", "p1_rank", "p2_rank",
    "h2h_diff", "h2h_total", "h2h_pct",
    "form_10_diff", "form_20_diff", "form_50_diff", "form_10_surf_diff",
    "p1_win_streak", "p2_win_streak",
    "p1_days_since", "p2_days_since",
    "p1_matches_90d", "p2_matches_90d",
    "ace_rate_diff", "first_in_diff", "first_won_diff", "second_won_diff", "bp_save_diff",
    "age_diff", "height_diff", "hand_matchup",
    "surface", "round", "tourney_level", "best_of", "is_atp",
    "implied_prob_p1", "ps_implied_p1",
]

# Simulation periods
BACKTEST_START = 2022
BACKTEST_END = 2024
LIVE_START = 2025

INITIAL_BANKROLL = 10000.0
MIN_EDGE_THRESHOLDS = [0.03, 0.05, 0.08, 0.10, 0.15]


# ============================================================
# LOAD
# ============================================================
def load_model_and_data():
    log.info("Loading model and features...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(FEATURES_PATH, low_memory=False)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")

    # Only matches with odds
    df = df[df["_odds_p1"].notna() & df["_odds_p2"].notna()].copy()

    # Sort by date
    df = df.sort_values("tourney_date").reset_index(drop=True)

    # Generate predictions
    X = df[FEATURE_COLS].values
    df["model_prob_p1"] = model.predict(X, num_iteration=model.best_iteration)
    df["model_prob_p2"] = 1.0 - df["model_prob_p1"]

    # Implied probabilities from market
    df["market_prob_p1"] = 1.0 / df["_odds_p1"]
    df["market_prob_p2"] = 1.0 / df["_odds_p2"]

    log.info(f"  {len(df):,} matches with odds and predictions")
    return model, df


# ============================================================
# BETTING STRATEGIES
# ============================================================
class FlatStake:
    """Bet a fixed percentage of initial bankroll on each qualifying bet."""
    name = "Flat (1% stake)"

    def __init__(self, bankroll, stake_pct=0.01):
        self.initial = bankroll
        self.stake_pct = stake_pct

    def get_stake(self, bankroll, odds, model_prob):
        return self.initial * self.stake_pct


class KellyCriterion:
    """Full Kelly: f* = (bp - q) / b where b=odds-1, p=model_prob, q=1-p.
    Capped at 10% of bankroll per bet, max stake $5000."""
    name = "Full Kelly"

    def get_stake(self, bankroll, odds, model_prob):
        b = odds - 1
        p = model_prob
        q = 1 - p
        f = (b * p - q) / b
        f = max(0, min(f, 0.10))  # Cap at 10% of bankroll
        stake = bankroll * f
        return min(stake, 5000)  # Hard cap per bet


class FractionalKelly:
    """Quarter Kelly — much safer, lower variance.
    Capped at max stake $2000."""
    name = "Quarter Kelly"

    def __init__(self, fraction=0.25):
        self.fraction = fraction

    def get_stake(self, bankroll, odds, model_prob):
        b = odds - 1
        p = model_prob
        q = 1 - p
        f = (b * p - q) / b
        f = max(0, min(f, 0.10)) * self.fraction
        stake = bankroll * f
        return min(stake, 2000)  # Hard cap per bet


# ============================================================
# SIMULATION ENGINE
# ============================================================
def simulate(df, strategy, min_edge=0.05, period_name="backtest", use_max_odds=False):
    """
    Walk forward through matches day by day.
    Place a bet when model edge > min_edge over market implied probability.
    """
    bankroll = INITIAL_BANKROLL
    peak_bankroll = bankroll

    bets = []
    daily_bankroll = []
    current_date = None

    odds_col_p1 = "_max_odds_p1" if use_max_odds else "_odds_p1"
    odds_col_p2 = "_max_odds_p2" if use_max_odds else "_odds_p2"

    for _, row in df.iterrows():
        d = row["tourney_date"]

        # Track daily bankroll
        if current_date != d:
            daily_bankroll.append({"date": d, "bankroll": bankroll})
            current_date = d

        if bankroll <= 0:
            break

        model_p1 = row["model_prob_p1"]
        model_p2 = row["model_prob_p2"]
        market_p1 = row["market_prob_p1"]
        market_p2 = row["market_prob_p2"]
        odds_p1 = row[odds_col_p1] if pd.notna(row.get(odds_col_p1)) else row["_odds_p1"]
        odds_p2 = row[odds_col_p2] if pd.notna(row.get(odds_col_p2)) else row["_odds_p2"]
        actual_p1_won = row["label"] == 1

        # Check edge on P1
        edge_p1 = model_p1 - market_p1
        # Check edge on P2
        edge_p2 = model_p2 - market_p2

        bet_on = None
        if edge_p1 > min_edge and edge_p1 >= edge_p2:
            bet_on = "p1"
            odds = odds_p1
            model_prob = model_p1
            won = actual_p1_won
        elif edge_p2 > min_edge:
            bet_on = "p2"
            odds = odds_p2
            model_prob = model_p2
            won = not actual_p1_won

        if bet_on and pd.notna(odds) and odds > 1:
            stake = strategy.get_stake(bankroll, odds, model_prob)
            stake = min(stake, bankroll)  # Can't bet more than you have

            if stake < 1:  # Skip micro bets
                continue

            if won:
                profit = stake * (odds - 1)
            else:
                profit = -stake

            bankroll += profit
            peak_bankroll = max(peak_bankroll, bankroll)

            bets.append({
                "date": d,
                "bet_on": bet_on,
                "odds": odds,
                "model_prob": model_prob,
                "market_prob": market_p1 if bet_on == "p1" else market_p2,
                "edge": edge_p1 if bet_on == "p1" else edge_p2,
                "stake": stake,
                "won": won,
                "profit": profit,
                "bankroll": bankroll,
                "drawdown": (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0,
            })

    # Final day
    daily_bankroll.append({"date": current_date, "bankroll": bankroll})

    return bets, daily_bankroll


# ============================================================
# ANALYSIS
# ============================================================
def analyze_results(bets, period_name, strategy_name, min_edge):
    if not bets:
        log.info(f"  No bets placed (edge threshold too high)")
        return None

    bdf = pd.DataFrame(bets)
    total_bets = len(bdf)
    wins = bdf["won"].sum()
    losses = total_bets - wins
    win_rate = wins / total_bets * 100

    total_staked = bdf["stake"].sum()
    total_profit = bdf["profit"].sum()
    roi = total_profit / total_staked * 100 if total_staked > 0 else 0
    final_bankroll = bdf["bankroll"].iloc[-1]
    bankroll_return = (final_bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100

    max_drawdown = bdf["drawdown"].max() * 100
    avg_odds = bdf["odds"].mean()
    avg_edge = bdf["edge"].mean() * 100
    avg_stake = bdf["stake"].mean()

    # Streak analysis
    streaks = []
    current_streak = 0
    for w in bdf["won"]:
        if not w:
            current_streak -= 1
        else:
            if current_streak < 0:
                streaks.append(current_streak)
            current_streak = 0
    if current_streak < 0:
        streaks.append(current_streak)
    worst_losing_streak = min(streaks) if streaks else 0

    # Monthly breakdown
    bdf["month"] = bdf["date"].dt.to_period("M")
    monthly = bdf.groupby("month").agg(
        bets=("profit", "count"),
        profit=("profit", "sum"),
        staked=("stake", "sum"),
        win_rate=("won", "mean"),
    )
    monthly["roi"] = monthly["profit"] / monthly["staked"] * 100
    profitable_months = (monthly["profit"] > 0).sum()
    total_months = len(monthly)

    # Sharpe-like ratio (monthly)
    if len(monthly) > 1 and monthly["profit"].std() > 0:
        sharpe = monthly["profit"].mean() / monthly["profit"].std() * np.sqrt(12)
    else:
        sharpe = 0

    results = {
        "period": period_name,
        "strategy": strategy_name,
        "min_edge": f"{min_edge:.0%}",
        "total_bets": total_bets,
        "wins": wins,
        "losses": losses,
        "win_rate": f"{win_rate:.1f}%",
        "total_staked": f"${total_staked:,.0f}",
        "total_profit": f"${total_profit:,.0f}",
        "roi": f"{roi:+.1f}%",
        "final_bankroll": f"${final_bankroll:,.0f}",
        "bankroll_return": f"{bankroll_return:+.1f}%",
        "max_drawdown": f"{max_drawdown:.1f}%",
        "avg_odds": f"{avg_odds:.2f}",
        "avg_edge": f"{avg_edge:.1f}%",
        "avg_stake": f"${avg_stake:.0f}",
        "worst_losing_streak": worst_losing_streak,
        "profitable_months": f"{profitable_months}/{total_months}",
        "sharpe_ratio": f"{sharpe:.2f}",
    }

    return results, bdf, monthly


def print_results(results):
    if results is None:
        return
    r, _, _ = results
    log.info(f"\n  {r['strategy']} | Edge > {r['min_edge']} | {r['period']}")
    log.info(f"  {'─'*50}")
    log.info(f"  Bets:            {r['total_bets']:>8}   (W:{r['wins']} / L:{r['losses']})")
    log.info(f"  Win Rate:        {r['win_rate']:>8}")
    log.info(f"  Avg Odds:        {r['avg_odds']:>8}")
    log.info(f"  Avg Edge:        {r['avg_edge']:>8}")
    log.info(f"  Total Staked:    {r['total_staked']:>8}")
    log.info(f"  Total Profit:    {r['total_profit']:>8}")
    log.info(f"  ROI:             {r['roi']:>8}")
    log.info(f"  Bankroll:        ${INITIAL_BANKROLL:,.0f} → {r['final_bankroll']}")
    log.info(f"  Max Drawdown:    {r['max_drawdown']:>8}")
    log.info(f"  Losing Streak:   {r['worst_losing_streak']:>8}")
    log.info(f"  Profitable Mo:   {r['profitable_months']:>8}")
    log.info(f"  Sharpe Ratio:    {r['sharpe_ratio']:>8}")


def print_monthly_table(monthly, label):
    log.info(f"\n  Monthly Breakdown ({label}):")
    log.info(f"  {'Month':>10} {'Bets':>6} {'Win%':>7} {'Profit':>10} {'ROI':>8}")
    log.info(f"  {'─'*45}")
    for period, row in monthly.iterrows():
        log.info(f"  {str(period):>10} {row['bets']:>6.0f} {row['win_rate']*100:>6.1f}% ${row['profit']:>+9,.0f} {row['roi']:>+7.1f}%")


# ============================================================
# MAIN
# ============================================================
def main():
    import time
    start = time.time()

    log.info("*" * 60)
    log.info("  TENNIS PREDICTOR — PHASE 3: SIMULATION ENGINE")
    log.info("*" * 60)

    model, df = load_model_and_data()

    # Split into backtest and live periods
    backtest = df[(df["year"] >= BACKTEST_START) & (df["year"] <= BACKTEST_END)].copy()
    live = df[df["year"] >= LIVE_START].copy()

    log.info(f"  Backtest period: {BACKTEST_START}-{BACKTEST_END} ({len(backtest):,} matches)")
    log.info(f"  Live period:     {LIVE_START}+ ({len(live):,} matches)")

    strategies = [
        FlatStake(INITIAL_BANKROLL, stake_pct=0.01),
        FractionalKelly(fraction=0.25),
        KellyCriterion(),
    ]

    all_results = []

    # ── BACKTEST (2022-2024) ──
    log.info(f"\n{'='*60}")
    log.info(f"BACKTEST: {BACKTEST_START}-{BACKTEST_END}")
    log.info(f"{'='*60}")

    best_result = None
    best_roi = -999

    for strategy in strategies:
        for min_edge in MIN_EDGE_THRESHOLDS:
            bets, daily = simulate(backtest, strategy, min_edge, "backtest")
            results = analyze_results(bets, f"{BACKTEST_START}-{BACKTEST_END}", strategy.name, min_edge)
            if results:
                print_results(results)
                r, bdf, monthly = results
                all_results.append(r)

                # Track best config
                roi_val = float(r["roi"].replace("+", "").replace("%", ""))
                total_bets = r["total_bets"]
                if roi_val > best_roi and total_bets >= 50:
                    best_roi = roi_val
                    best_result = (strategy, min_edge, r)

    # Show best config
    if best_result:
        strat, edge, r = best_result
        log.info(f"\n{'='*60}")
        log.info(f"BEST BACKTEST CONFIG")
        log.info(f"{'='*60}")
        log.info(f"  Strategy: {strat.name}")
        log.info(f"  Min Edge: {edge:.0%}")
        log.info(f"  ROI: {r['roi']}  |  Bets: {r['total_bets']}  |  Bankroll: {r['final_bankroll']}")

        # Run this best config on monthly detail
        bets, daily = simulate(backtest, strat, edge, "backtest")
        results = analyze_results(bets, f"{BACKTEST_START}-{BACKTEST_END}", strat.name, edge)
        if results:
            _, bdf, monthly = results
            print_monthly_table(monthly, f"Best Config Backtest {BACKTEST_START}-{BACKTEST_END}")

            # Save detailed bets
            bdf.to_csv(SIM_OUTPUT_DIR / "backtest_bets.csv", index=False)

        # ── LIVE TEST with best config ──
        log.info(f"\n{'='*60}")
        log.info(f"LIVE TEST: {LIVE_START}+ (using best backtest config)")
        log.info(f"{'='*60}")

        bets_live, daily_live = simulate(live, strat, edge, "live")
        results_live = analyze_results(bets_live, f"{LIVE_START}+", strat.name, edge)
        if results_live:
            print_results(results_live)
            r_live, bdf_live, monthly_live = results_live
            all_results.append(r_live)
            print_monthly_table(monthly_live, f"Live {LIVE_START}+")
            bdf_live.to_csv(SIM_OUTPUT_DIR / "live_bets.csv", index=False)

    # ── Also test all strategies on live ──
    log.info(f"\n{'='*60}")
    log.info(f"ALL STRATEGIES ON LIVE ({LIVE_START}+)")
    log.info(f"{'='*60}")

    for strategy in strategies:
        for min_edge in [0.05, 0.10, 0.15]:
            bets, daily = simulate(live, strategy, min_edge, "live")
            results = analyze_results(bets, f"{LIVE_START}+", strategy.name, min_edge)
            if results:
                print_results(results)

    # ── Save summary ──
    summary_path = SIM_OUTPUT_DIR / "simulation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\nResults saved to {SIM_OUTPUT_DIR}/")

    # ── Save daily bankroll for charting ──
    if best_result:
        bets_all, daily_all = simulate(
            df[df["year"] >= BACKTEST_START],
            best_result[0], best_result[1], "full"
        )
        if daily_all:
            pd.DataFrame(daily_all).to_csv(SIM_OUTPUT_DIR / "daily_bankroll.csv", index=False)

    elapsed = time.time() - start
    log.info(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log.info("NEXT: Run Phase 4 — backend API")


if __name__ == "__main__":
    main()
