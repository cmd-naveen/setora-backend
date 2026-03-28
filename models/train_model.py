"""
Tennis Prediction Model — Phase 2: Training
=============================================
Trains a LightGBM model to predict match outcomes and find value bets.

Strategy:
  - Training data: 2010-2021 (matches with betting odds era)
  - Validation: 2022-2024
  - Test: 2025-2026
  - Pre-2010 data: Only for Elo rating initialization
  - Features: Elo, rankings, H2H, recent form, serve stats, physical, context
  - Target: P(player_1 wins) — player order randomized to avoid label leakage

Usage:
  cd models/
  python train_model.py
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss,
    classification_report, roc_auc_score
)
from sklearn.calibration import calibration_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("train")

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data-pipeline" / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

MATCHES_FILE = DATA_DIR / "matches_clean.csv"
ELO_HISTORY_FILE = DATA_DIR / "elo_history.csv"
PLAYERS_FILE = DATA_DIR / "players.csv"

# ============================================================
# CONFIG
# ============================================================
TRAIN_START = 2010
TRAIN_END = 2021
VAL_START = 2022
VAL_END = 2024
TEST_START = 2025

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ============================================================
# STEP 1: LOAD & PREPARE DATA
# ============================================================
def load_data():
    log.info("Loading data...")
    matches = pd.read_csv(MATCHES_FILE, low_memory=False)
    elo_hist = pd.read_csv(ELO_HISTORY_FILE, low_memory=False)
    players = pd.read_csv(PLAYERS_FILE, low_memory=False)

    # Parse dates
    matches["tourney_date"] = pd.to_datetime(matches["tourney_date"], errors="coerce")
    matches["year"] = matches["tourney_date"].dt.year
    elo_hist["date"] = pd.to_datetime(elo_hist["date"], errors="coerce")

    # Filter to completed matches only
    if "is_completed" in matches.columns:
        matches = matches[matches["is_completed"] == True].copy()

    log.info(f"  Total matches: {len(matches):,}")
    log.info(f"  2010+: {(matches['year'] >= 2010).sum():,}")
    log.info(f"  With odds: {matches['B365W'].notna().sum():,}")

    return matches, elo_hist, players


# ============================================================
# STEP 2: BUILD ELO LOOKUP
# ============================================================
def build_elo_lookup(elo_hist):
    """Build a lookup: (date, player_id) → (overall_elo, surface_elo)."""
    log.info("Building Elo lookup from history...")

    # Latest Elo per player (overall) — rolling state
    # We'll build per-match Elo by indexing into elo_history
    # Create a dict: player_id → list of (date, elo_after)
    player_elo = defaultdict(list)
    surface_elo = defaultdict(lambda: defaultdict(list))

    for _, row in elo_hist.iterrows():
        d = row["date"]
        w, l = row["winner_id"], row["loser_id"]
        s = row.get("surface", "")

        player_elo[w].append((d, row["w_elo_after"]))
        player_elo[l].append((d, row["l_elo_after"]))

        if pd.notna(s) and s:
            surface_elo[w][s].append((d, row["w_elo_after"]))
            surface_elo[l][s].append((d, row["l_elo_after"]))

    log.info(f"  Elo lookup built for {len(player_elo):,} players")
    return player_elo, surface_elo


def get_elo_at_date(player_elo_list, date, default=1500.0):
    """Get most recent Elo rating before a given date."""
    if not player_elo_list:
        return default
    # Find last entry before date
    best = default
    for d, elo in player_elo_list:
        if d < date:
            best = elo
        else:
            break
    return best


# ============================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================
def compute_h2h(matches_before_2010_plus):
    """Build running H2H records from all historical matches."""
    log.info("Computing head-to-head records...")
    h2h = defaultdict(lambda: [0, 0])  # (p1, p2) → [p1_wins, p2_wins]

    # Sort by date
    sorted_matches = matches_before_2010_plus.sort_values("tourney_date")

    # Store H2H state at each match for lookup
    h2h_at_match = {}

    for idx, row in sorted_matches.iterrows():
        w = row.get("winner_id")
        l = row.get("loser_id")
        if pd.isna(w) or pd.isna(l):
            continue
        w, l = int(w), int(l)
        key = (min(w, l), max(w, l))

        # Record current state BEFORE this match
        if w < l:
            h2h_at_match[idx] = (h2h[key][0], h2h[key][1])
        else:
            h2h_at_match[idx] = (h2h[key][1], h2h[key][0])

        # Update after match
        if w == key[0]:
            h2h[key][0] += 1
        else:
            h2h[key][1] += 1

    return h2h_at_match


def compute_rolling_form(matches_all):
    """Compute rolling win rates per player (last N matches)."""
    log.info("Computing rolling form stats...")
    sorted_m = matches_all.sort_values("tourney_date").reset_index(drop=True)

    player_results = defaultdict(list)  # player_id → list of (date, won, surface)
    form_at_match = {}

    for idx, row in sorted_m.iterrows():
        w = row.get("winner_id")
        l = row.get("loser_id")
        s = row.get("surface", "")
        d = row["tourney_date"]
        if pd.isna(w) or pd.isna(l):
            continue
        w, l = int(w), int(l)

        # Record form BEFORE this match
        w_results = player_results[w]
        l_results = player_results[l]

        def calc_form(results, surface, windows=[10, 20, 50]):
            out = {}
            recent = results[-max(windows):] if results else []
            for n in windows:
                last_n = recent[-n:]
                out[f"form_{n}"] = np.mean([r[1] for r in last_n]) if last_n else 0.5
                # Surface-specific
                surf_n = [r for r in last_n if r[2] == surface]
                out[f"form_{n}_surf"] = np.mean([r[1] for r in surf_n]) if surf_n else 0.5
            # Streak
            streak = 0
            for r in reversed(recent):
                if r[1] == 1:
                    streak += 1
                else:
                    break
            out["win_streak"] = streak
            # Days since last match
            out["days_since_last"] = (d - recent[-1][0]).days if recent else 90
            out["matches_last_90d"] = sum(1 for r in recent if (d - r[0]).days <= 90)
            return out

        w_form = calc_form(w_results, s)
        l_form = calc_form(l_results, s)
        form_at_match[idx] = (w_form, l_form)

        # Update
        player_results[w].append((d, 1, s))
        player_results[l].append((d, 0, s))

    return form_at_match


def compute_serve_stats_rolling(matches_all):
    """Rolling serve stats per player (last 20 matches with stats)."""
    log.info("Computing rolling serve stats...")
    sorted_m = matches_all.sort_values("tourney_date").reset_index(drop=True)

    player_serve = defaultdict(list)  # player_id → list of stat dicts
    serve_at_match = {}

    serve_cols_w = ["w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_bpSaved", "w_bpFaced"]
    serve_cols_l = ["l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_bpSaved", "l_bpFaced"]

    for idx, row in sorted_m.iterrows():
        w = row.get("winner_id")
        l = row.get("loser_id")
        if pd.isna(w) or pd.isna(l):
            continue
        w, l = int(w), int(l)

        # Stats BEFORE this match
        def avg_serve(hist, n=20):
            recent = hist[-n:]
            if not recent:
                return {}
            out = {}
            for key in recent[0]:
                vals = [r[key] for r in recent if pd.notna(r.get(key))]
                out[key] = np.mean(vals) if vals else np.nan
            return out

        serve_at_match[idx] = (avg_serve(player_serve[w]), avg_serve(player_serve[l]))

        # Record this match's stats
        has_w_stats = pd.notna(row.get("w_svpt")) and row.get("w_svpt", 0) > 0
        has_l_stats = pd.notna(row.get("l_svpt")) and row.get("l_svpt", 0) > 0

        if has_w_stats:
            svpt = row["w_svpt"]
            player_serve[w].append({
                "ace_rate": row["w_ace"] / svpt if svpt else 0,
                "df_rate": row["w_df"] / svpt if svpt else 0,
                "1st_in_pct": row["w_1stIn"] / svpt if svpt else 0,
                "1st_won_pct": row["w_1stWon"] / row["w_1stIn"] if row.get("w_1stIn", 0) > 0 else 0,
                "2nd_won_pct": row["w_2ndWon"] / (svpt - row["w_1stIn"]) if (svpt - row.get("w_1stIn", 0)) > 0 else 0,
                "bp_save_pct": row["w_bpSaved"] / row["w_bpFaced"] if row.get("w_bpFaced", 0) > 0 else 0,
            })
        if has_l_stats:
            svpt = row["l_svpt"]
            player_serve[l].append({
                "ace_rate": row["l_ace"] / svpt if svpt else 0,
                "df_rate": row["l_df"] / svpt if svpt else 0,
                "1st_in_pct": row["l_1stIn"] / svpt if svpt else 0,
                "1st_won_pct": row["l_1stWon"] / row["l_1stIn"] if row.get("l_1stIn", 0) > 0 else 0,
                "2nd_won_pct": row["l_2ndWon"] / (svpt - row["l_1stIn"]) if (svpt - row.get("l_1stIn", 0)) > 0 else 0,
                "bp_save_pct": row["l_bpSaved"] / row["l_bpFaced"] if row.get("l_bpFaced", 0) > 0 else 0,
            })

    return serve_at_match


# ============================================================
# STEP 4: BUILD FEATURE MATRIX
# ============================================================
def build_features(matches, elo_hist, players):
    """Build feature matrix from matches 2010+, using all history for rolling stats."""
    log.info("=" * 60)
    log.info("BUILDING FEATURE MATRIX")
    log.info("=" * 60)

    # Sort all matches by date for rolling computations
    all_matches = matches.sort_values("tourney_date").reset_index(drop=True)

    # Compute rolling features over ALL matches (including pre-2010 for history)
    h2h_at_match = compute_h2h(all_matches)
    form_at_match = compute_rolling_form(all_matches)
    serve_at_match = compute_serve_stats_rolling(all_matches)

    # Build Elo lookup
    player_elo, surface_elo = build_elo_lookup(elo_hist)

    # Build player info lookup
    player_info = {}
    if not players.empty:
        for _, p in players.iterrows():
            pid = p.get("player_id")
            if pd.notna(pid):
                player_info[int(pid)] = {
                    "hand": p.get("hand", "U"),
                    "height": p.get("height"),
                }

    # Filter to 2010+ for feature rows
    mask_2010 = all_matches["year"] >= TRAIN_START
    indices_2010 = all_matches[mask_2010].index.tolist()

    log.info(f"Building features for {len(indices_2010):,} matches (2010+)...")

    rows = []
    for idx in indices_2010:
        row = all_matches.loc[idx]
        w_id = row.get("winner_id")
        l_id = row.get("loser_id")
        if pd.isna(w_id) or pd.isna(l_id):
            continue
        w_id, l_id = int(w_id), int(l_id)
        d = row["tourney_date"]
        surface = row.get("surface", "Hard")
        year = row["year"]

        # Randomize player order to avoid label leakage
        if np.random.random() < 0.5:
            p1, p2 = w_id, l_id
            label = 1  # p1 wins
            p1_name, p2_name = row.get("winner_name", ""), row.get("loser_name", "")
            p1_rank, p2_rank = row.get("winner_rank"), row.get("loser_rank")
            p1_rank_pts, p2_rank_pts = row.get("winner_rank_points"), row.get("loser_rank_points")
            p1_age, p2_age = row.get("winner_age"), row.get("loser_age")
            p1_ht, p2_ht = row.get("winner_ht"), row.get("loser_ht")
            p1_seed, p2_seed = row.get("winner_seed"), row.get("loser_seed")
            odds_p1, odds_p2 = row.get("B365W"), row.get("B365L")
            ps_p1, ps_p2 = row.get("PSW"), row.get("PSL")
            max_p1, max_p2 = row.get("MaxW"), row.get("MaxL")
            avg_p1, avg_p2 = row.get("AvgW"), row.get("AvgL")
            # H2H
            h2h_rec = h2h_at_match.get(idx, (0, 0))
            if w_id < l_id:
                p1_h2h, p2_h2h = h2h_rec
            else:
                p2_h2h, p1_h2h = h2h_rec
            # Form
            w_form, l_form = form_at_match.get(idx, ({}, {}))
            p1_form, p2_form = w_form, l_form
            # Serve
            w_serve, l_serve = serve_at_match.get(idx, ({}, {}))
            p1_serve, p2_serve = w_serve, l_serve
        else:
            p1, p2 = l_id, w_id
            label = 0  # p1 loses
            p1_name, p2_name = row.get("loser_name", ""), row.get("winner_name", "")
            p1_rank, p2_rank = row.get("loser_rank"), row.get("winner_rank")
            p1_rank_pts, p2_rank_pts = row.get("loser_rank_points"), row.get("winner_rank_points")
            p1_age, p2_age = row.get("loser_age"), row.get("winner_age")
            p1_ht, p2_ht = row.get("loser_ht"), row.get("winner_ht")
            p1_seed, p2_seed = row.get("loser_seed"), row.get("winner_seed")
            odds_p1, odds_p2 = row.get("B365L"), row.get("B365W")
            ps_p1, ps_p2 = row.get("PSL"), row.get("PSW")
            max_p1, max_p2 = row.get("MaxL"), row.get("MaxW")
            avg_p1, avg_p2 = row.get("AvgL"), row.get("AvgW")
            # H2H
            h2h_rec = h2h_at_match.get(idx, (0, 0))
            if l_id < w_id:
                p1_h2h, p2_h2h = h2h_rec
            else:
                p2_h2h, p1_h2h = h2h_rec
            # Form
            w_form, l_form = form_at_match.get(idx, ({}, {}))
            p1_form, p2_form = l_form, w_form
            # Serve
            w_serve, l_serve = serve_at_match.get(idx, ({}, {}))
            p1_serve, p2_serve = l_serve, w_serve

        # --- Elo features ---
        p1_elo = get_elo_at_date(player_elo.get(p1, []), d)
        p2_elo = get_elo_at_date(player_elo.get(p2, []), d)
        p1_elo_surf = get_elo_at_date(surface_elo.get(p1, {}).get(surface, []), d)
        p2_elo_surf = get_elo_at_date(surface_elo.get(p2, {}).get(surface, []), d)

        # --- Ranking features ---
        rank1 = p1_rank if pd.notna(p1_rank) else 500
        rank2 = p2_rank if pd.notna(p2_rank) else 500
        rpts1 = p1_rank_pts if pd.notna(p1_rank_pts) else 0
        rpts2 = p2_rank_pts if pd.notna(p2_rank_pts) else 0

        # --- Player info ---
        p1_info = player_info.get(p1, {})
        p2_info = player_info.get(p2, {})
        ht1 = p1_ht if pd.notna(p1_ht) else (p1_info.get("height") or np.nan)
        ht2 = p2_ht if pd.notna(p2_ht) else (p2_info.get("height") or np.nan)
        age1 = p1_age if pd.notna(p1_age) else np.nan
        age2 = p2_age if pd.notna(p2_age) else np.nan

        # Handedness matchup
        h1 = str(p1_info.get("hand", "U")).upper()
        h2 = str(p2_info.get("hand", "U")).upper()
        hand_matchup = 0  # same
        if h1 == "R" and h2 == "L":
            hand_matchup = 1
        elif h1 == "L" and h2 == "R":
            hand_matchup = -1

        # --- Surface encoding ---
        surface_map = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}
        surface_enc = surface_map.get(surface, 0)

        # --- Round encoding ---
        round_map = {"F": 7, "SF": 6, "QF": 5, "R16": 4, "R32": 3, "R64": 2, "R128": 1, "RR": 3}
        round_enc = round_map.get(row.get("round", ""), 3)

        # --- Tourney level encoding ---
        level_map = {"G": 5, "M": 4, "A": 3, "P": 4, "D": 3, "F": 5, "C": 2}
        level_enc = level_map.get(row.get("tourney_level", ""), 3)

        # --- Implied probability from odds ---
        def safe_odds(v):
            try:
                v = float(v)
                return v if v > 1 else np.nan
            except (ValueError, TypeError):
                return np.nan

        odds_p1 = safe_odds(odds_p1)
        odds_p2 = safe_odds(odds_p2)
        ps_p1 = safe_odds(ps_p1)
        ps_p2 = safe_odds(ps_p2)
        max_p1 = safe_odds(max_p1)
        max_p2 = safe_odds(max_p2)
        avg_p1 = safe_odds(avg_p1)
        avg_p2 = safe_odds(avg_p2)

        impl_prob_p1 = 1.0 / odds_p1 if pd.notna(odds_p1) else np.nan
        impl_prob_p2 = 1.0 / odds_p2 if pd.notna(odds_p2) else np.nan

        # Pinnacle implied (sharper line)
        ps_impl_p1 = 1.0 / ps_p1 if pd.notna(ps_p1) else np.nan

        # --- Build feature dict ---
        feat = {
            "label": label,
            "year": year,
            "tourney_date": d,
            "p1_id": p1,
            "p2_id": p2,

            # Elo
            "elo_diff": p1_elo - p2_elo,
            "elo_surf_diff": p1_elo_surf - p2_elo_surf,
            "p1_elo": p1_elo,
            "p2_elo": p2_elo,

            # Rankings
            "rank_diff": rank2 - rank1,  # positive = p1 ranked higher (lower number)
            "log_rank_ratio": np.log1p(rank2) - np.log1p(rank1),
            "rank_pts_diff": rpts1 - rpts2,
            "p1_rank": rank1,
            "p2_rank": rank2,

            # H2H
            "h2h_diff": p1_h2h - p2_h2h,
            "h2h_total": p1_h2h + p2_h2h,
            "h2h_pct": p1_h2h / (p1_h2h + p2_h2h) if (p1_h2h + p2_h2h) > 0 else 0.5,

            # Form
            "form_10_diff": p1_form.get("form_10", 0.5) - p2_form.get("form_10", 0.5),
            "form_20_diff": p1_form.get("form_20", 0.5) - p2_form.get("form_20", 0.5),
            "form_50_diff": p1_form.get("form_50", 0.5) - p2_form.get("form_50", 0.5),
            "form_10_surf_diff": p1_form.get("form_10_surf", 0.5) - p2_form.get("form_10_surf", 0.5),
            "p1_win_streak": p1_form.get("win_streak", 0),
            "p2_win_streak": p2_form.get("win_streak", 0),
            "p1_days_since": p1_form.get("days_since_last", 90),
            "p2_days_since": p2_form.get("days_since_last", 90),
            "p1_matches_90d": p1_form.get("matches_last_90d", 0),
            "p2_matches_90d": p2_form.get("matches_last_90d", 0),

            # Serve stats (rolling averages)
            "ace_rate_diff": (p1_serve.get("ace_rate", np.nan) or np.nan) - (p2_serve.get("ace_rate", np.nan) or np.nan) if p1_serve and p2_serve else np.nan,
            "first_in_diff": (p1_serve.get("1st_in_pct", np.nan) or np.nan) - (p2_serve.get("1st_in_pct", np.nan) or np.nan) if p1_serve and p2_serve else np.nan,
            "first_won_diff": (p1_serve.get("1st_won_pct", np.nan) or np.nan) - (p2_serve.get("1st_won_pct", np.nan) or np.nan) if p1_serve and p2_serve else np.nan,
            "second_won_diff": (p1_serve.get("2nd_won_pct", np.nan) or np.nan) - (p2_serve.get("2nd_won_pct", np.nan) or np.nan) if p1_serve and p2_serve else np.nan,
            "bp_save_diff": (p1_serve.get("bp_save_pct", np.nan) or np.nan) - (p2_serve.get("bp_save_pct", np.nan) or np.nan) if p1_serve and p2_serve else np.nan,

            # Physical
            "age_diff": (age1 - age2) if pd.notna(age1) and pd.notna(age2) else np.nan,
            "height_diff": (ht1 - ht2) if pd.notna(ht1) and pd.notna(ht2) else np.nan,
            "hand_matchup": hand_matchup,

            # Context
            "surface": surface_enc,
            "round": round_enc,
            "tourney_level": level_enc,
            "best_of": row.get("best_of", 3),
            "is_atp": 1 if row.get("tour") == "atp" else 0,

            # Market odds (as features for calibration)
            "implied_prob_p1": impl_prob_p1,
            "ps_implied_p1": ps_impl_p1,

            # Raw odds for ROI calculation (not features)
            "_odds_p1": odds_p1,
            "_odds_p2": odds_p2,
            "_max_odds_p1": max_p1,
            "_max_odds_p2": max_p2,
            "_avg_odds_p1": avg_p1,
            "_avg_odds_p2": avg_p2,
        }
        rows.append(feat)

    df = pd.DataFrame(rows)
    log.info(f"Feature matrix: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ============================================================
# STEP 5: TRAIN / VALIDATE / TEST SPLIT
# ============================================================
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


def split_data(df):
    train = df[(df["year"] >= TRAIN_START) & (df["year"] <= TRAIN_END)].copy()
    val = df[(df["year"] >= VAL_START) & (df["year"] <= VAL_END)].copy()
    test = df[df["year"] >= TEST_START].copy()

    log.info(f"  Train: {len(train):,} ({TRAIN_START}-{TRAIN_END})")
    log.info(f"  Val:   {len(val):,} ({VAL_START}-{VAL_END})")
    log.info(f"  Test:  {len(test):,} ({TEST_START}+)")

    return train, val, test


def compute_sample_weights(train):
    """Time-decay weights: recent matches get more weight."""
    years = train["year"].values
    max_year = years.max()
    # Exponential decay: half-life of 4 years
    weights = np.exp(-0.17 * (max_year - years))
    # Normalize
    weights = weights / weights.mean()
    return weights


# ============================================================
# STEP 6: TRAIN MODEL
# ============================================================
def train_model(train, val):
    log.info("=" * 60)
    log.info("TRAINING LIGHTGBM MODEL")
    log.info("=" * 60)

    X_train = train[FEATURE_COLS].values
    y_train = train["label"].values
    X_val = val[FEATURE_COLS].values
    y_val = val["label"].values

    weights = compute_sample_weights(train)

    train_data = lgb.Dataset(X_train, label=y_train, weight=weights, feature_name=FEATURE_COLS)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLS, reference=train_data)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 7,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "seed": RANDOM_SEED,
    }

    callbacks = [
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=50),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        valid_names=["val"],
        callbacks=callbacks,
    )

    log.info(f"Best iteration: {model.best_iteration}")
    return model


# ============================================================
# STEP 7: EVALUATE
# ============================================================
def evaluate(model, df, name="Test"):
    log.info(f"\n{'='*60}")
    log.info(f"EVALUATION: {name}")
    log.info(f"{'='*60}")

    X = df[FEATURE_COLS].values
    y_true = df["label"].values
    y_prob = model.predict(X, num_iteration=model.best_iteration)
    y_pred = (y_prob >= 0.5).astype(int)

    # Core metrics
    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    log.info(f"  Accuracy:    {acc:.4f} ({acc*100:.1f}%)")
    log.info(f"  Log Loss:    {ll:.4f}")
    log.info(f"  Brier Score: {brier:.4f}")
    log.info(f"  AUC:         {auc:.4f}")

    # Calibration
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    log.info(f"\n  Calibration (predicted → actual):")
    for pt, pp in zip(prob_pred, prob_true):
        bar = "█" * int(pp * 40)
        log.info(f"    {pp:.2f} → {pt:.2f}  {bar}")

    # Compare against bookmaker odds
    has_odds = df["_odds_p1"].notna()
    if has_odds.sum() > 100:
        log.info(f"\n  --- Value Betting Analysis ({has_odds.sum():,} matches with odds) ---")

        odds_df = df[has_odds].copy()
        y_true_o = odds_df["label"].values
        y_prob_o = model.predict(odds_df[FEATURE_COLS].values, num_iteration=model.best_iteration)
        market_impl = 1.0 / odds_df["_odds_p1"].values

        # Bookmaker accuracy
        bk_pred = (market_impl >= 0.5).astype(int)
        bk_acc = accuracy_score(y_true_o, bk_pred)
        bk_ll = log_loss(y_true_o, np.clip(market_impl, 0.01, 0.99))

        log.info(f"  Bookmaker accuracy:  {bk_acc:.4f} ({bk_acc*100:.1f}%)")
        log.info(f"  Bookmaker log loss:  {bk_ll:.4f}")
        log.info(f"  Model accuracy:      {accuracy_score(y_true_o, (y_prob_o >= 0.5).astype(int)):.4f}")
        log.info(f"  Model log loss:      {log_loss(y_true_o, y_prob_o):.4f}")

        # Simulated ROI — bet when model disagrees with market
        for threshold in [0.05, 0.08, 0.10, 0.15]:
            edge = y_prob_o - market_impl
            bet_mask = edge > threshold  # Model thinks p1 is undervalued
            if bet_mask.sum() < 10:
                continue
            bet_odds = odds_df["_odds_p1"].values[bet_mask]
            bet_results = y_true_o[bet_mask]
            pnl = np.sum(bet_results * (bet_odds - 1) - (1 - bet_results))
            roi = pnl / bet_mask.sum() * 100
            win_rate = bet_results.mean() * 100
            log.info(f"  Edge>{threshold:.0%}: {bet_mask.sum():,} bets, win {win_rate:.1f}%, ROI {roi:+.1f}%")

            # Also bet on p2 when model thinks p2 undervalued
            edge_p2 = (1 - y_prob_o) - (1 - market_impl)
            bet_mask_p2 = edge_p2 > threshold
            if bet_mask_p2.sum() >= 10:
                bet_odds_p2 = odds_df["_odds_p2"].values[bet_mask_p2]
                bet_results_p2 = (1 - y_true_o[bet_mask_p2])
                pnl_p2 = np.sum(bet_results_p2 * (bet_odds_p2 - 1) - (1 - bet_results_p2))
                total_bets = bet_mask.sum() + bet_mask_p2.sum()
                total_pnl = pnl + pnl_p2
                combined_roi = total_pnl / total_bets * 100
                log.info(f"          + p2 bets: {total_bets:,} total bets, combined ROI {combined_roi:+.1f}%")

    return {"accuracy": acc, "log_loss": ll, "brier": brier, "auc": auc}


# ============================================================
# STEP 8: FEATURE IMPORTANCE
# ============================================================
def show_feature_importance(model):
    log.info(f"\n{'='*60}")
    log.info("FEATURE IMPORTANCE")
    log.info(f"{'='*60}")

    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(FEATURE_COLS, importance), key=lambda x: -x[1])

    max_imp = feat_imp[0][1] if feat_imp else 1
    for name, imp in feat_imp:
        bar = "█" * int(imp / max_imp * 30)
        log.info(f"  {name:25s} {imp:10.0f}  {bar}")


# ============================================================
# STEP 9: SAVE MODEL
# ============================================================
def save_model(model, metrics, feature_cols):
    log.info(f"\n{'='*60}")
    log.info("SAVING MODEL")
    log.info(f"{'='*60}")

    # Save LightGBM model
    model_path = OUTPUT_DIR / "tennis_model.txt"
    model.save_model(str(model_path))
    log.info(f"  Model: {model_path}")

    # Save as pickle too for easy loading
    pickle_path = OUTPUT_DIR / "tennis_model.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(model, f)
    log.info(f"  Pickle: {pickle_path}")

    # Save metadata
    meta = {
        "created": datetime.now().isoformat(),
        "train_years": f"{TRAIN_START}-{TRAIN_END}",
        "val_years": f"{VAL_START}-{VAL_END}",
        "test_years": f"{TEST_START}+",
        "features": feature_cols,
        "best_iteration": model.best_iteration,
        "metrics": metrics,
        "model_type": "lightgbm",
    }
    meta_path = OUTPUT_DIR / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    log.info(f"  Metadata: {meta_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    import time
    start = time.time()

    log.info("*" * 60)
    log.info("  TENNIS PREDICTOR — PHASE 2: MODEL TRAINING")
    log.info("*" * 60)

    # Load
    matches, elo_hist, players = load_data()

    # Build features
    feature_df = build_features(matches, elo_hist, players)

    # Save feature matrix for debugging/reuse
    feature_df.to_csv(OUTPUT_DIR / "features.csv", index=False)
    log.info(f"Feature matrix saved to {OUTPUT_DIR / 'features.csv'}")

    # Split
    train, val, test = split_data(feature_df)

    # Train
    model = train_model(train, val)

    # Evaluate
    val_metrics = evaluate(model, val, "Validation (2022-2024)")
    test_metrics = evaluate(model, test, "Test (2025-2026)")

    # Feature importance
    show_feature_importance(model)

    # Save
    save_model(model, {"val": val_metrics, "test": test_metrics}, FEATURE_COLS)

    elapsed = time.time() - start
    log.info(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log.info("NEXT: Run Phase 3 — simulation engine")


if __name__ == "__main__":
    main()
