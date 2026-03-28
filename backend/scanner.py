"""
Auto-Scanner Engine — The brain of the automated betting system.

Scans upcoming matches with live odds, runs model predictions,
identifies value bets, auto-places paper bets on high-confidence picks,
and tracks performance for self-evaluation.

Can run on a schedule (every 30 min) or be triggered manually.
"""

import json
import pickle
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from odds_fetcher import fetch_live_odds, get_cached_odds, match_odds_to_players, load_api_key

BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"               # small CSVs committed to repo
MODEL_DIR = PROJECT_DIR / "models" / "output"
MATCHES_DB = DATA_DIR / "matches.db"       # SQLite H2H (replaces 145MB CSV)
CACHE_DIR = BASE_DIR / "cache"

# ============================================================
# LOAD MODEL & DATA
# ============================================================

def load_model():
    with open(MODEL_DIR / "tennis_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "model_meta.json") as f:
        meta = json.load(f)
    return model, meta


def load_player_data():
    active = []
    if (CACHE_DIR / "active_players.json").exists():
        with open(CACHE_DIR / "active_players.json") as f:
            active = json.load(f)

    active_dict = {p["player_id"]: p for p in active}

    name_to_id = {}
    if (CACHE_DIR / "name_to_id.json").exists():
        with open(CACHE_DIR / "name_to_id.json") as f:
            name_to_id = json.load(f)
            # Keys are strings, values might be ints
            name_to_id = {k: int(v) for k, v in name_to_id.items()}

    elo = pd.read_csv(DATA_DIR / "elo_ratings.csv", low_memory=False)
    elo_dict = elo.set_index("player_id").to_dict("index")

    stats = pd.read_csv(DATA_DIR / "player_stats.csv", low_memory=False)
    stats_dict = stats.set_index("player_id").to_dict("index")

    players = pd.read_csv(DATA_DIR / "players.csv", low_memory=False)
    players["full_name"] = (players["name_first"].fillna("") + " " + players["name_last"].fillna("")).str.strip()

    return {
        "active": active,
        "active_dict": active_dict,
        "name_to_id": name_to_id,
        "elo_dict": elo_dict,
        "stats_dict": stats_dict,
        "players": players,
    }


# ============================================================
# FEATURE BUILDER (standalone, mirrors app.py logic)
# ============================================================

def build_features_for_match(p1_id, p2_id, surface, tour, best_of,
                              tourney_level, round_name, odds_p1, odds_p2,
                              data, feature_cols):
    """Build feature vector for a match. Returns numpy array."""
    elo1 = data["elo_dict"].get(p1_id, {})
    elo2 = data["elo_dict"].get(p2_id, {})
    stats1 = data["stats_dict"].get(p1_id, {})
    stats2 = data["stats_dict"].get(p2_id, {})
    ap1 = data["active_dict"].get(p1_id, {})
    ap2 = data["active_dict"].get(p2_id, {})

    p1_row = data["players"][data["players"]["player_id"] == p1_id]
    p2_row = data["players"][data["players"]["player_id"] == p2_id]
    p1_info = p1_row.iloc[0] if not p1_row.empty else {}
    p2_info = p2_row.iloc[0] if not p2_row.empty else {}

    p1_elo = elo1.get("elo_overall", 1500)
    p2_elo = elo2.get("elo_overall", 1500)
    surf_lower = surface.lower() if surface else "hard"
    surface_key = f"elo_{surf_lower}" if surf_lower in ["hard", "clay", "grass", "carpet"] else "elo_hard"
    p1_elo_surf = elo1.get(surface_key, 1500)
    p2_elo_surf = elo2.get(surface_key, 1500)

    p1_rank = ap1.get("rank") or 200
    p2_rank = ap2.get("rank") or 200
    p1_rank_pts = ap1.get("rank_points") or 0
    p2_rank_pts = ap2.get("rank_points") or 0

    # H2H from SQLite
    h2h_p1 = h2h_p2 = 0
    if MATCHES_DB.exists():
        try:
            conn = sqlite3.connect(MATCHES_DB)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM matches WHERE winner_id=? AND loser_id=?", (p1_id, p2_id))
            h2h_p1 = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM matches WHERE winner_id=? AND loser_id=?", (p2_id, p1_id))
            h2h_p2 = cur.fetchone()[0]
            conn.close()
        except Exception:
            pass
    h2h_total = h2h_p1 + h2h_p2
    h2h_pct = h2h_p1 / h2h_total if h2h_total > 0 else 0.5

    form1_10 = stats1.get("form_last_10", 0.5)
    form1_20 = stats1.get("form_last_20", 0.5)
    form2_10 = stats2.get("form_last_10", 0.5)
    form2_20 = stats2.get("form_last_20", 0.5)

    surf_key = f"win_rate_{surf_lower}"
    form1_surf = stats1.get(surf_key, 0.5)
    form2_surf = stats2.get(surf_key, 0.5)

    ace1 = stats1.get("avg_ace_rate", np.nan)
    ace2 = stats2.get("avg_ace_rate", np.nan)
    sv1 = stats1.get("avg_1st_serve_pct", np.nan)
    sv2 = stats2.get("avg_1st_serve_pct", np.nan)
    svw1 = stats1.get("avg_1st_serve_win_pct", np.nan)
    svw2 = stats2.get("avg_1st_serve_win_pct", np.nan)

    age1 = age2 = None
    if isinstance(p1_info, pd.Series) and pd.notna(p1_info.get("dob")):
        try:
            age1 = (datetime.now() - pd.to_datetime(p1_info["dob"])).days / 365.25
        except Exception:
            pass
    if isinstance(p2_info, pd.Series) and pd.notna(p2_info.get("dob")):
        try:
            age2 = (datetime.now() - pd.to_datetime(p2_info["dob"])).days / 365.25
        except Exception:
            pass

    ht1 = p1_info.get("height") if isinstance(p1_info, pd.Series) else None
    ht2 = p2_info.get("height") if isinstance(p2_info, pd.Series) else None

    h1 = str(p1_info.get("hand", "U")).upper() if isinstance(p1_info, pd.Series) else "U"
    h2 = str(p2_info.get("hand", "U")).upper() if isinstance(p2_info, pd.Series) else "U"
    hand_matchup = 0
    if h1 == "R" and h2 == "L": hand_matchup = 1
    elif h1 == "L" and h2 == "R": hand_matchup = -1

    surface_map = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}
    round_map = {"F": 7, "SF": 6, "QF": 5, "R16": 4, "R32": 3, "R64": 2, "R128": 1, "RR": 3}
    level_map = {"G": 5, "M": 4, "A": 3, "P": 4, "D": 3, "F": 5, "C": 2}

    impl_p1 = 1.0 / odds_p1 if odds_p1 and odds_p1 > 1 else np.nan

    features = {
        "elo_diff": p1_elo - p2_elo,
        "elo_surf_diff": p1_elo_surf - p2_elo_surf,
        "p1_elo": p1_elo,
        "p2_elo": p2_elo,
        "rank_diff": p2_rank - p1_rank,
        "log_rank_ratio": np.log1p(p2_rank) - np.log1p(p1_rank),
        "rank_pts_diff": p1_rank_pts - p2_rank_pts,
        "p1_rank": p1_rank,
        "p2_rank": p2_rank,
        "h2h_diff": h2h_p1 - h2h_p2,
        "h2h_total": h2h_total,
        "h2h_pct": h2h_pct,
        "form_10_diff": form1_10 - form2_10,
        "form_20_diff": form1_20 - form2_20,
        "form_50_diff": form1_20 - form2_20,
        "form_10_surf_diff": form1_surf - form2_surf,
        "p1_win_streak": ap1.get("win_streak", 0),
        "p2_win_streak": ap2.get("win_streak", 0),
        "p1_days_since": ap1.get("days_since_match", 14),
        "p2_days_since": ap2.get("days_since_match", 14),
        "p1_matches_90d": min(ap1.get("matches_6m", 0), 30),
        "p2_matches_90d": min(ap2.get("matches_6m", 0), 30),
        "ace_rate_diff": (ace1 - ace2) if pd.notna(ace1) and pd.notna(ace2) else np.nan,
        "first_in_diff": (sv1 - sv2) if pd.notna(sv1) and pd.notna(sv2) else np.nan,
        "first_won_diff": (svw1 - svw2) if pd.notna(svw1) and pd.notna(svw2) else np.nan,
        "second_won_diff": np.nan,
        "bp_save_diff": np.nan,
        "age_diff": (age1 - age2) if age1 and age2 else np.nan,
        "height_diff": (float(ht1) - float(ht2)) if pd.notna(ht1) and pd.notna(ht2) else np.nan,
        "hand_matchup": hand_matchup,
        "surface": surface_map.get(surface, 0),
        "round": round_map.get(round_name, 3),
        "tourney_level": level_map.get(tourney_level, 3),
        "best_of": best_of,
        "is_atp": 1 if tour == "atp" else 0,
        "implied_prob_p1": impl_p1,
        "ps_implied_p1": impl_p1,
    }

    return np.array([[features.get(c, np.nan) for c in feature_cols]])


# ============================================================
# SCANNER — CORE
# ============================================================

class ScanResult:
    """Result of scanning a single match."""
    def __init__(self, match_data, p1_id, p2_id, p1_name, p2_name,
                 prob_p1, prob_p2, odds_p1, odds_p2, surface, tour):
        self.match_data = match_data
        self.p1_id = p1_id
        self.p2_id = p2_id
        self.p1_name = p1_name
        self.p2_name = p2_name
        self.prob_p1 = prob_p1
        self.prob_p2 = prob_p2
        self.odds_p1 = odds_p1
        self.odds_p2 = odds_p2
        self.surface = surface
        self.tour = tour

        # Derived
        self.impl_p1 = 1.0 / odds_p1 if odds_p1 > 1 else 0
        self.impl_p2 = 1.0 / odds_p2 if odds_p2 > 1 else 0
        self.edge_p1 = prob_p1 - self.impl_p1
        self.edge_p2 = prob_p2 - self.impl_p2
        self.confidence = max(prob_p1, prob_p2)
        self.predicted_winner = p1_name if prob_p1 > prob_p2 else p2_name
        self.predicted_winner_id = p1_id if prob_p1 > prob_p2 else p2_id

        # Best value bet
        if self.edge_p1 > self.edge_p2 and self.edge_p1 > 0:
            self.best_bet = "p1"
            self.best_edge = self.edge_p1
            self.best_odds = odds_p1
            self.best_prob = prob_p1
            self.bet_on_name = p1_name
        elif self.edge_p2 > 0:
            self.best_bet = "p2"
            self.best_edge = self.edge_p2
            self.best_odds = odds_p2
            self.best_prob = prob_p2
            self.bet_on_name = p2_name
        else:
            self.best_bet = None
            self.best_edge = 0
            self.best_odds = 0
            self.best_prob = 0
            self.bet_on_name = ""

        # Value rating
        if self.best_edge > 0.15:
            self.rating = "STRONG"
        elif self.best_edge > 0.10:
            self.rating = "GOOD"
        elif self.best_edge > 0.05:
            self.rating = "MILD"
        else:
            self.rating = "NO VALUE"

        # Kelly stake (quarter Kelly, capped at 5%)
        if self.best_bet and self.best_odds > 1:
            b = self.best_odds - 1
            p = self.best_prob
            q = 1 - p
            kelly = (b * p - q) / b if b > 0 else 0
            self.kelly_pct = max(0, min(kelly * 0.25, 0.05)) * 100  # Quarter Kelly, max 5%
        else:
            self.kelly_pct = 0

    def to_dict(self):
        return {
            "p1_id": self.p1_id,
            "p2_id": self.p2_id,
            "p1_name": self.p1_name,
            "p2_name": self.p2_name,
            "prob_p1": round(self.prob_p1 * 100, 1),
            "prob_p2": round(self.prob_p2 * 100, 1),
            "odds_p1": self.odds_p1,
            "odds_p2": self.odds_p2,
            "edge_p1": round(self.edge_p1 * 100, 1),
            "edge_p2": round(self.edge_p2 * 100, 1),
            "confidence": round(self.confidence * 100, 1),
            "predicted_winner": self.predicted_winner,
            "best_bet": self.best_bet,
            "best_edge": round(self.best_edge * 100, 1),
            "best_odds": self.best_odds,
            "bet_on_name": self.bet_on_name,
            "rating": self.rating,
            "kelly_pct": round(self.kelly_pct, 2),
            "surface": self.surface,
            "tour": self.tour,
            "commence_time": self.match_data.get("commence_time", ""),
        }


def run_scan(use_cached_odds=True, min_confidence=0.55, min_edge=0.03):
    """
    Run the full scanner pipeline:
    1. Fetch/load odds
    2. Match players to our database
    3. Run model predictions
    4. Rank by confidence and edge
    5. Return scan results
    """
    print(f"\n{'='*50}")
    print(f"SCANNER RUN — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")

    # Load model and data
    model, meta = load_model()
    feature_cols = meta["features"]
    best_iter = meta["best_iteration"]
    data = load_player_data()

    # Get odds
    if use_cached_odds:
        odds_data = get_cached_odds()
        if odds_data.get("is_stale", True):
            print("Cached odds are stale, fetching fresh...")
            odds_data = fetch_live_odds()
    else:
        odds_data = fetch_live_odds()

    if odds_data.get("error"):
        print(f"Odds error: {odds_data['error']}")

    matches = odds_data.get("matches", [])
    if not matches:
        print("No matches with odds available.")
        return {
            "scan_time": datetime.now().isoformat(),
            "matches_scanned": 0,
            "value_bets": [],
            "all_predictions": [],
            "high_confidence": [],
        }

    # Match odds to players
    matched = match_odds_to_players(matches, data["name_to_id"], data["active_dict"])
    both_matched = [m for m in matched if m.get("both_matched")]
    print(f"Matches with odds: {len(matches)}")
    print(f"Both players matched: {len(both_matched)}")

    # Run predictions
    results = []
    for m in both_matched:
        p1_id = m["home_player_id"]
        p2_id = m["away_player_id"]
        p1_name = m.get("home", "")
        p2_name = m.get("away", "")
        tour = m.get("tour", "atp")
        best_of = 3  # Default
        surface = "Hard"  # Default, could be enriched from tournament data

        odds_p1 = m.get("best_odds", {}).get("avg_home", 0)
        odds_p2 = m.get("best_odds", {}).get("avg_away", 0)

        if not odds_p1 or not odds_p2 or odds_p1 <= 1 or odds_p2 <= 1:
            continue

        try:
            X = build_features_for_match(
                p1_id, p2_id, surface, tour, best_of,
                "A", "R32", odds_p1, odds_p2,
                data, feature_cols
            )
            prob_p1 = float(model.predict(X, num_iteration=best_iter)[0])
            prob_p2 = 1.0 - prob_p1

            result = ScanResult(
                m, p1_id, p2_id, p1_name, p2_name,
                prob_p1, prob_p2, odds_p1, odds_p2, surface, tour
            )
            results.append(result)

        except Exception as e:
            print(f"  Error predicting {p1_name} vs {p2_name}: {e}")

    # Filter and sort
    all_predictions = sorted(results, key=lambda r: r.confidence, reverse=True)
    value_bets = [r for r in results if r.best_edge > min_edge and r.confidence > min_confidence]
    value_bets.sort(key=lambda r: r.best_edge, reverse=True)

    high_confidence = [r for r in results if r.confidence > 0.70]
    high_confidence.sort(key=lambda r: r.confidence, reverse=True)

    # Auto-bet candidates: confidence > 80% AND edge > 10%
    auto_bet_candidates = [r for r in results if r.confidence > 0.80 and r.best_edge > 0.10]

    print(f"\nResults:")
    print(f"  Total predictions: {len(results)}")
    print(f"  Value bets (edge>{min_edge*100}%): {len(value_bets)}")
    print(f"  High confidence (>70%): {len(high_confidence)}")
    print(f"  Auto-bet candidates (>80% conf, >10% edge): {len(auto_bet_candidates)}")

    for r in value_bets[:5]:
        print(f"  {r.rating}: {r.p1_name} vs {r.p2_name} — "
              f"Pred: {r.predicted_winner} ({r.confidence*100:.1f}%) — "
              f"Edge: {r.best_edge*100:.1f}% @ {r.best_odds} — "
              f"Kelly: {r.kelly_pct:.1f}%")

    scan_result = {
        "scan_time": datetime.now().isoformat(),
        "matches_scanned": len(results),
        "value_bets": [r.to_dict() for r in value_bets],
        "all_predictions": [r.to_dict() for r in all_predictions],
        "high_confidence": [r.to_dict() for r in high_confidence],
        "auto_bet_candidates": [r.to_dict() for r in auto_bet_candidates],
    }

    # Save scan results
    with open(CACHE_DIR / "last_scan.json", "w") as f:
        json.dump(scan_result, f, indent=2)

    return scan_result


# ============================================================
# AUTO-BET ENGINE
# ============================================================

def auto_place_bets(scan_results, bankroll=10000, max_daily_bets=10):
    """
    Automatically place paper bets on high-confidence picks.

    Rules:
    - Only bet on matches with confidence > 80% AND edge > 10%
    - Use Quarter Kelly for stake sizing
    - Max 10 auto-bets per day
    - Track all auto-bets separately for evaluation
    """
    AUTO_BETS_FILE = CACHE_DIR / "auto_bets.json"
    auto_bets = []
    if AUTO_BETS_FILE.exists():
        with open(AUTO_BETS_FILE) as f:
            auto_bets = json.load(f)

    # Count today's bets
    today = datetime.now().strftime("%Y-%m-%d")
    today_bets = sum(1 for b in auto_bets if b.get("date") == today)

    candidates = scan_results.get("auto_bet_candidates", [])
    new_bets = []

    for c in candidates:
        if today_bets + len(new_bets) >= max_daily_bets:
            break

        # Check if already bet on this matchup today
        match_key = f"{c['p1_id']}_{c['p2_id']}_{today}"
        if any(b.get("match_key") == match_key for b in auto_bets):
            continue

        # Calculate stake using Kelly
        stake = round(bankroll * (c["kelly_pct"] / 100), 2)
        stake = max(10, min(stake, bankroll * 0.05))  # Min $10, max 5% of bankroll

        bet = {
            "id": len(auto_bets) + len(new_bets) + 1,
            "date": today,
            "created_at": datetime.now().isoformat(),
            "match_key": match_key,
            "p1_id": c["p1_id"],
            "p2_id": c["p2_id"],
            "p1_name": c["p1_name"],
            "p2_name": c["p2_name"],
            "bet_on": c["best_bet"],
            "bet_on_name": c["bet_on_name"],
            "odds": c["best_odds"],
            "stake": stake,
            "model_prob": c["confidence"] / 100,
            "edge": c["best_edge"] / 100,
            "kelly_pct": c["kelly_pct"],
            "rating": c["rating"],
            "surface": c.get("surface", ""),
            "tour": c.get("tour", ""),
            "status": "pending",
            "won": None,
            "profit": None,
            "auto": True,
        }
        new_bets.append(bet)

    if new_bets:
        auto_bets.extend(new_bets)
        with open(AUTO_BETS_FILE, "w") as f:
            json.dump(auto_bets, f, indent=2, default=str)
        print(f"\nAuto-placed {len(new_bets)} paper bets")
        for b in new_bets:
            print(f"  {b['bet_on_name']} @ {b['odds']} — ${b['stake']} — Edge: {b['edge']*100:.1f}%")

    return new_bets


# ============================================================
# SELF-EVALUATION
# ============================================================

def evaluate_performance():
    """
    Evaluate the auto-betting system's performance.
    Returns accuracy, ROI, calibration, and recommendations.
    """
    AUTO_BETS_FILE = CACHE_DIR / "auto_bets.json"
    PAPER_BETS_FILE = BASE_DIR / "paper_bets.json"

    all_bets = []

    if AUTO_BETS_FILE.exists():
        with open(AUTO_BETS_FILE) as f:
            all_bets.extend(json.load(f))

    if PAPER_BETS_FILE.exists():
        with open(PAPER_BETS_FILE) as f:
            all_bets.extend(json.load(f))

    settled = [b for b in all_bets if b.get("status") == "settled"]
    pending = [b for b in all_bets if b.get("status") == "pending"]

    if not settled:
        return {
            "total_bets": len(all_bets),
            "settled": 0,
            "pending": len(pending),
            "message": "No settled bets yet. Place some bets and settle them to see performance.",
        }

    # Basic stats
    wins = sum(1 for b in settled if b.get("won"))
    losses = len(settled) - wins
    total_staked = sum(b.get("stake", 0) for b in settled)
    total_profit = sum(b.get("profit", 0) for b in settled)
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0

    # By confidence bucket
    buckets = {"50-60%": [], "60-70%": [], "70-80%": [], "80-90%": [], "90%+": []}
    for b in settled:
        conf = b.get("model_prob", 0.5) * 100
        if conf >= 90: buckets["90%+"].append(b)
        elif conf >= 80: buckets["80-90%"].append(b)
        elif conf >= 70: buckets["70-80%"].append(b)
        elif conf >= 60: buckets["60-70%"].append(b)
        else: buckets["50-60%"].append(b)

    calibration = {}
    for bucket_name, bets in buckets.items():
        if bets:
            wins_b = sum(1 for b in bets if b.get("won"))
            actual_rate = wins_b / len(bets) * 100
            avg_conf = sum(b.get("model_prob", 0.5) for b in bets) / len(bets) * 100
            profit_b = sum(b.get("profit", 0) for b in bets)
            staked_b = sum(b.get("stake", 0) for b in bets)
            roi_b = (profit_b / staked_b * 100) if staked_b > 0 else 0
            calibration[bucket_name] = {
                "count": len(bets),
                "wins": wins_b,
                "actual_win_rate": round(actual_rate, 1),
                "avg_confidence": round(avg_conf, 1),
                "calibration_error": round(actual_rate - avg_conf, 1),
                "profit": round(profit_b, 2),
                "roi": round(roi_b, 1),
            }

    # By edge bucket
    edge_buckets = {"5-10%": [], "10-15%": [], "15-20%": [], "20%+": []}
    for b in settled:
        edge = abs(b.get("edge", 0)) * 100
        if edge >= 20: edge_buckets["20%+"].append(b)
        elif edge >= 15: edge_buckets["15-20%"].append(b)
        elif edge >= 10: edge_buckets["10-15%"].append(b)
        else: edge_buckets["5-10%"].append(b)

    edge_analysis = {}
    for bucket_name, bets in edge_buckets.items():
        if bets:
            wins_b = sum(1 for b in bets if b.get("won"))
            profit_b = sum(b.get("profit", 0) for b in bets)
            staked_b = sum(b.get("stake", 0) for b in bets)
            edge_analysis[bucket_name] = {
                "count": len(bets),
                "win_rate": round(wins_b / len(bets) * 100, 1),
                "profit": round(profit_b, 2),
                "roi": round((profit_b / staked_b * 100) if staked_b > 0 else 0, 1),
            }

    # Auto vs manual comparison
    auto_settled = [b for b in settled if b.get("auto")]

    auto_perf = {}
    if auto_settled:
        auto_wins = sum(1 for b in auto_settled if b.get("won"))
        auto_staked = sum(b.get("stake", 0) for b in auto_settled)
        auto_profit = sum(b.get("profit", 0) for b in auto_settled)
        auto_perf = {
            "count": len(auto_settled),
            "win_rate": round(auto_wins / len(auto_settled) * 100, 1),
            "roi": round((auto_profit / auto_staked * 100) if auto_staked > 0 else 0, 1),
            "profit": round(auto_profit, 2),
        }

    # Streak analysis
    current_streak = 0
    streak_type = None
    for b in reversed(settled):
        if streak_type is None:
            streak_type = "W" if b.get("won") else "L"
            current_streak = 1
        elif (b.get("won") and streak_type == "W") or (not b.get("won") and streak_type == "L"):
            current_streak += 1
        else:
            break

    # Recommendations
    recommendations = []
    if len(settled) < 30:
        recommendations.append("Need at least 30 settled bets for reliable evaluation.")
    if roi < -10:
        recommendations.append("Negative ROI detected. Consider raising the minimum edge threshold.")
    if calibration.get("80-90%", {}).get("calibration_error", 0) < -10:
        recommendations.append("Model is overconfident in 80-90% range. Consider adjusting confidence threshold.")
    if edge_analysis.get("5-10%", {}).get("roi", 0) < 0:
        recommendations.append("Low-edge bets (5-10%) are losing money. Raise minimum edge to 10%+.")

    return {
        "total_bets": len(all_bets),
        "settled": len(settled),
        "pending": len(pending),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / len(settled) * 100, 1),
        "total_staked": round(total_staked, 2),
        "total_profit": round(total_profit, 2),
        "roi": round(roi, 1),
        "current_streak": f"{current_streak}{streak_type}" if streak_type else "0",
        "calibration": calibration,
        "edge_analysis": edge_analysis,
        "auto_performance": auto_perf,
        "recommendations": recommendations,
    }


if __name__ == "__main__":
    results = run_scan(use_cached_odds=False)
    if results.get("auto_bet_candidates"):
        auto_place_bets(results)
    perf = evaluate_performance()
    print(f"\nPerformance: {json.dumps(perf, indent=2)}")
