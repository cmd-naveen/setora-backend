"""
Tennis Prediction API — Phase 4: Backend
==========================================
FastAPI server that serves predictions, player data, and paper trading.

Endpoints:
  GET  /                          — Health check
  GET  /api/players               — Search players by name
  GET  /api/players/{id}          — Player profile (stats, Elo, recent form)
  POST /api/predict               — Predict match outcome
  GET  /api/predictions/today     — Today's value bets (if odds feed connected)
  POST /api/paper-bets            — Record a paper bet
  GET  /api/paper-bets            — List all paper bets
  GET  /api/paper-bets/summary    — Paper trading P&L summary
  PATCH /api/paper-bets/{id}      — Settle a paper bet (mark won/lost)
  GET  /api/h2h/{p1_id}/{p2_id}   — Head-to-head record

Usage:
  cd backend/
  uvicorn app:app --reload --port 8000
"""

import json
import os
import pickle
import logging
import sqlite3
import threading
import subprocess
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from odds_fetcher import fetch_live_odds, get_cached_odds, save_api_key, load_api_key
from scanner import run_scan, auto_place_bets, evaluate_performance
from live_pulse import manager as pulse_manager, start_live_simulation
from news_fetcher import fetch_news, get_cached_news, get_live_match_news

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("api")

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"          # small CSVs + matches.db live here
MODEL_DIR = PROJECT_DIR / "models" / "output"
MATCHES_DB = DATA_DIR / "matches.db"  # SQLite H2H database (replaces 145MB CSV)

# ============================================================
# LOAD DATA & MODEL AT STARTUP  (all wrapped — never crash on Railway)
# ============================================================
CACHE_DIR = BASE_DIR / "cache"
_STARTUP_ERRORS: list = []

log.info(f"BASE_DIR={BASE_DIR}  DATA_DIR={DATA_DIR}  MODEL_DIR={MODEL_DIR}")

# Model
MODEL = None
MODEL_META: dict = {"features": [], "best_iteration": 0, "model_type": "unknown"}
FEATURE_COLS: list = []
BEST_ITERATION = 0
try:
    with open(MODEL_DIR / "tennis_model.pkl", "rb") as f:
        MODEL = pickle.load(f)
    with open(MODEL_DIR / "model_meta.json") as f:
        MODEL_META = json.load(f)
    FEATURE_COLS = MODEL_META.get("features", [])
    BEST_ITERATION = MODEL_META.get("best_iteration", 0)
    log.info(f"Model loaded: {MODEL_META.get('model_type')} ({len(FEATURE_COLS)} features)")
except Exception as e:
    _STARTUP_ERRORS.append(f"model: {e}")
    log.error(f"Model load failed (predictions disabled): {e}")

# Player data
PLAYERS = pd.DataFrame()
try:
    PLAYERS = pd.read_csv(DATA_DIR / "players.csv", low_memory=False)
    PLAYERS["full_name"] = (PLAYERS["name_first"].fillna("") + " " + PLAYERS["name_last"].fillna("")).str.strip()
    PLAYERS["player_id"] = PLAYERS["player_id"].astype(int)
    log.info(f"Players loaded: {len(PLAYERS):,}")
except Exception as e:
    _STARTUP_ERRORS.append(f"players.csv: {e}")
    log.error(f"players.csv load failed: {e}")

# Active players index
ACTIVE_PLAYERS: list = []
ACTIVE_DICT: dict = {}
try:
    if (CACHE_DIR / "active_players.json").exists():
        with open(CACHE_DIR / "active_players.json") as f:
            ACTIVE_PLAYERS = json.load(f)
        ACTIVE_DICT = {p["player_id"]: p for p in ACTIVE_PLAYERS}
        log.info(f"Active players: {len(ACTIVE_PLAYERS)}")
except Exception as e:
    _STARTUP_ERRORS.append(f"active_players.json: {e}")
    log.error(f"active_players.json load failed: {e}")

# Recent matches feed
RECENT_MATCHES: list = []
try:
    if (CACHE_DIR / "recent_matches.json").exists():
        with open(CACHE_DIR / "recent_matches.json") as f:
            RECENT_MATCHES = json.load(f)
        log.info(f"Recent matches: {len(RECENT_MATCHES)}")
except Exception as e:
    _STARTUP_ERRORS.append(f"recent_matches.json: {e}")
    log.error(f"recent_matches.json load failed: {e}")

# Name→ID lookup
NAME_TO_ID: dict = {}
try:
    if (CACHE_DIR / "name_to_id.json").exists():
        with open(CACHE_DIR / "name_to_id.json") as f:
            NAME_TO_ID = {k: int(v) for k, v in json.load(f).items()}
except Exception as e:
    log.warning(f"name_to_id.json load failed: {e}")

# Elo ratings
ELO = pd.DataFrame()
ELO_DICT: dict = {}
try:
    ELO = pd.read_csv(DATA_DIR / "elo_ratings.csv", low_memory=False)
    ELO["player_id"] = ELO["player_id"].astype(int)
    ELO_DICT = ELO.set_index("player_id").to_dict("index")
    log.info(f"Elo ratings: {len(ELO):,}")
except Exception as e:
    _STARTUP_ERRORS.append(f"elo_ratings.csv: {e}")
    log.error(f"elo_ratings.csv load failed: {e}")

# Player stats
STATS = pd.DataFrame()
STATS_DICT: dict = {}
try:
    STATS = pd.read_csv(DATA_DIR / "player_stats.csv", low_memory=False)
    STATS["player_id"] = STATS["player_id"].astype(int)
    STATS_DICT = STATS.set_index("player_id").to_dict("index")
    log.info(f"Player stats: {len(STATS):,}")
except Exception as e:
    _STARTUP_ERRORS.append(f"player_stats.csv: {e}")
    log.error(f"player_stats.csv load failed: {e}")

# Paper bets storage
PAPER_BETS_FILE = BASE_DIR / "paper_bets.json"
PAPER_BETS: List[dict] = []
try:
    if PAPER_BETS_FILE.exists():
        with open(PAPER_BETS_FILE) as f:
            PAPER_BETS = json.load(f)
except Exception as e:
    log.warning(f"paper_bets.json load failed: {e}")

log.info(f"Startup complete. Errors: {_STARTUP_ERRORS if _STARTUP_ERRORS else 'none'}")

# ============================================================
# APP
# ============================================================
async def _news_refresh_loop():
    """Background task: refresh news cache every 30 minutes."""
    import asyncio
    await asyncio.sleep(10)
    while True:
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, fetch_news)
            log.info(f"News refreshed: {result.get('item_count', 0)} items")
        except Exception as e:
            log.error(f"News refresh failed: {e}")
        await asyncio.sleep(1800)  # 30 min


async def _odds_refresh_loop():
    """Background task: refresh live odds every 2 hours."""
    import asyncio
    await asyncio.sleep(30)
    while True:
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, fetch_live_odds)
            matches = result.get("matches", []) if isinstance(result, dict) else result
            log.info(f"Odds refreshed: {len(matches)} matches")
        except Exception as e:
            log.error(f"Odds refresh failed: {e}")
        await asyncio.sleep(7200)  # 2 hours


async def _daily_data_refresh_loop():
    """Background task: refresh player/match data once per day."""
    import asyncio
    await asyncio.sleep(60)  # wait 1 min after startup
    while True:
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: subprocess.run(
                [sys.executable, str(BASE_DIR / "refresh_data.py")],
                capture_output=True, timeout=300
            ))
            log.info("Daily data refresh completed")
        except Exception as e:
            log.error(f"Daily data refresh failed: {e}")
        await asyncio.sleep(86400)  # 24 hours


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    try:
        asyncio.create_task(start_live_simulation(app))
        log.info("Live simulation started")
    except Exception as e:
        log.error(f"Failed to start live simulation: {e}")
    asyncio.create_task(_news_refresh_loop())
    asyncio.create_task(_odds_refresh_loop())
    asyncio.create_task(_daily_data_refresh_loop())
    yield

app = FastAPI(
    title="Tennis Predictor API",
    description="Tennis match prediction & paper betting API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — production origins + localhost for dev
_default_origins = "https://setora.pro,https://www.setora.pro,https://setora.pages.dev,http://localhost:5173,http://localhost:4173"
_ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", _default_origins).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SCHEMAS
# ============================================================
class PredictRequest(BaseModel):
    player1_id: int
    player2_id: int
    surface: str = "Hard"  # Hard, Clay, Grass
    tourney_level: str = "A"  # G=Grand Slam, M=Masters, A=ATP/WTA, C=Challenger
    round: str = "R32"  # F, SF, QF, R16, R32, R64, R128
    best_of: int = 3
    tour: str = "atp"  # atp or wta
    odds_p1: Optional[float] = None  # Bookmaker odds for player 1
    odds_p2: Optional[float] = None  # Bookmaker odds for player 2


class PaperBetRequest(BaseModel):
    player1_id: int
    player2_id: int
    player1_name: str
    player2_name: str
    bet_on: str  # "p1" or "p2"
    odds: float
    stake: float
    model_prob: float
    edge: float
    surface: str = ""
    tournament: str = ""
    notes: str = ""


class SettleBetRequest(BaseModel):
    won: bool


# ============================================================
# HELPERS
# ============================================================
def get_player_info(player_id: int):
    # Prefer active player data (has rank, recent form, etc.)
    ap = ACTIVE_DICT.get(player_id)
    if ap:
        return {
            "player_id": player_id,
            "name": ap["name"],
            "hand": ap.get("hand", "U"),
            "height": ap.get("height"),
            "country": ap.get("country", ""),
            "dob": ap.get("dob"),
            "tour": ap.get("tour", ""),
            "rank": ap.get("rank"),
            "rank_points": ap.get("rank_points"),
            "is_active": True,
            "last_match_date": ap.get("last_match_date"),
            "days_since_match": ap.get("days_since_match", 0),
            "elo": {
                "overall": ap.get("elo_overall", 1500),
                "hard": ap.get("elo_hard", 1500),
                "clay": ap.get("elo_clay", 1500),
                "grass": ap.get("elo_grass", 1500),
            },
            "stats": {
                "total_matches": ap.get("total_matches", 0),
                "win_rate": ap.get("win_rate", 0),
                "win_rate_hard": ap.get("win_rate_hard", 0),
                "win_rate_clay": ap.get("win_rate_clay", 0),
                "win_rate_grass": ap.get("win_rate_grass", 0),
                "form_6m": ap.get("form_6m", 0),
                "form_last_10": ap.get("form_last_10", 0),
                "win_streak": ap.get("win_streak", 0),
                "matches_6m": ap.get("matches_6m", 0),
            },
            "recent_matches": ap.get("recent_matches", [])[:10],
        }

    # Fallback to base player data (retired/inactive)
    row = PLAYERS[PLAYERS["player_id"] == player_id]
    if row.empty:
        return None
    p = row.iloc[0]
    elo = ELO_DICT.get(player_id, {})
    stats = STATS_DICT.get(player_id, {})
    return {
        "player_id": player_id,
        "name": p["full_name"],
        "hand": p.get("hand", "U"),
        "height": int(p["height"]) if pd.notna(p.get("height")) else None,
        "country": p.get("ioc", ""),
        "dob": str(p.get("dob", "")) if pd.notna(p.get("dob")) else None,
        "tour": p.get("tour", ""),
        "rank": None,
        "rank_points": None,
        "is_active": False,
        "elo": {
            "overall": round(elo.get("elo_overall", 1500), 1),
            "hard": round(elo.get("elo_hard", 1500), 1),
            "clay": round(elo.get("elo_clay", 1500), 1),
            "grass": round(elo.get("elo_grass", 1500), 1),
        },
        "stats": {
            "total_matches": int(stats.get("total_matches", 0)),
            "win_rate": round(stats.get("win_rate", 0) * 100, 1),
        },
    }


def get_h2h(p1_id: int, p2_id: int):
    if not MATCHES_DB.exists():
        return {"p1_wins": 0, "p2_wins": 0, "total": 0, "matches": []}
    conn = sqlite3.connect(MATCHES_DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM matches WHERE (winner_id=? AND loser_id=?) OR (winner_id=? AND loser_id=?)"
        " ORDER BY tourney_date DESC LIMIT 20",
        (p1_id, p2_id, p2_id, p1_id),
    )
    rows = cur.fetchall()
    conn.close()

    p1_wins = sum(1 for r in rows if int(r["winner_id"]) == p1_id)
    p2_wins = len(rows) - p1_wins
    matches = [
        {
            "date": r["tourney_date"] or "",
            "tournament": r["tourney_name"] or "",
            "surface": r["surface"] or "",
            "round": r["round"] or "",
            "winner": r["winner_name"] or "",
            "score": r["score"] or "",
        }
        for r in rows
    ]
    return {"p1_wins": p1_wins, "p2_wins": p2_wins, "total": p1_wins + p2_wins, "matches": matches}


def build_features(p1_id: int, p2_id: int, req: PredictRequest):
    """Build feature vector for a match prediction."""
    elo1 = ELO_DICT.get(p1_id, {})
    elo2 = ELO_DICT.get(p2_id, {})
    stats1 = STATS_DICT.get(p1_id, {})
    stats2 = STATS_DICT.get(p2_id, {})

    p1_info = PLAYERS[PLAYERS["player_id"] == p1_id].iloc[0] if not PLAYERS[PLAYERS["player_id"] == p1_id].empty else {}
    p2_info = PLAYERS[PLAYERS["player_id"] == p2_id].iloc[0] if not PLAYERS[PLAYERS["player_id"] == p2_id].empty else {}

    # Elo
    p1_elo = elo1.get("elo_overall", 1500)
    p2_elo = elo2.get("elo_overall", 1500)
    surface_key = f"elo_{req.surface.lower()}" if req.surface.lower() in ["hard", "clay", "grass", "carpet"] else "elo_hard"
    p1_elo_surf = elo1.get(surface_key, 1500)
    p2_elo_surf = elo2.get(surface_key, 1500)

    # Rankings from active player data
    ap1 = ACTIVE_DICT.get(p1_id, {})
    ap2 = ACTIVE_DICT.get(p2_id, {})
    p1_rank = ap1.get("rank") or 200
    p2_rank = ap2.get("rank") or 200
    p1_rank_pts = ap1.get("rank_points") or 0
    p2_rank_pts = ap2.get("rank_points") or 0

    # H2H
    h2h = get_h2h(p1_id, p2_id)
    h2h_total = h2h["total"]
    h2h_pct = h2h["p1_wins"] / h2h_total if h2h_total > 0 else 0.5

    # Form
    form1_10 = stats1.get("form_last_10", 0.5)
    form1_20 = stats1.get("form_last_20", 0.5)
    form2_10 = stats2.get("form_last_10", 0.5)
    form2_20 = stats2.get("form_last_20", 0.5)

    # Surface win rates as form proxy
    surf_key = f"win_rate_{req.surface.lower()}"
    form1_surf = stats1.get(surf_key, 0.5)
    form2_surf = stats2.get(surf_key, 0.5)

    # Serve stats
    ace1 = stats1.get("avg_ace_rate", np.nan)
    ace2 = stats2.get("avg_ace_rate", np.nan)
    sv1 = stats1.get("avg_1st_serve_pct", np.nan)
    sv2 = stats2.get("avg_1st_serve_pct", np.nan)
    svw1 = stats1.get("avg_1st_serve_win_pct", np.nan)
    svw2 = stats2.get("avg_1st_serve_win_pct", np.nan)

    # Physical
    age1 = None
    age2 = None
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
    if h1 == "R" and h2 == "L":
        hand_matchup = 1
    elif h1 == "L" and h2 == "R":
        hand_matchup = -1

    # Context encodings
    surface_map = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}
    round_map = {"F": 7, "SF": 6, "QF": 5, "R16": 4, "R32": 3, "R64": 2, "R128": 1, "RR": 3}
    level_map = {"G": 5, "M": 4, "A": 3, "P": 4, "D": 3, "F": 5, "C": 2}

    # Implied prob from odds
    impl_p1 = 1.0 / req.odds_p1 if req.odds_p1 and req.odds_p1 > 1 else np.nan
    ps_impl_p1 = impl_p1  # Use same odds as Pinnacle proxy

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
        "h2h_diff": h2h["p1_wins"] - h2h["p2_wins"],
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
        "surface": surface_map.get(req.surface, 0),
        "round": round_map.get(req.round, 3),
        "tourney_level": level_map.get(req.tourney_level, 3),
        "best_of": req.best_of,
        "is_atp": 1 if req.tour == "atp" else 0,
        "implied_prob_p1": impl_p1,
        "ps_implied_p1": ps_impl_p1,
    }

    return np.array([[features[c] for c in FEATURE_COLS]])


def save_paper_bets():
    with open(PAPER_BETS_FILE, "w") as f:
        json.dump(PAPER_BETS, f, indent=2, default=str)


# ============================================================
# DATA REFRESH
# ============================================================
REFRESH_STATE: dict = {"running": False, "last_refresh": None, "last_status": "idle", "message": ""}


def _do_refresh():
    global RECENT_MATCHES, ACTIVE_PLAYERS, ACTIVE_DICT
    REFRESH_STATE["running"] = True
    REFRESH_STATE["message"] = "Downloading and processing latest data..."
    try:
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "refresh_data.py")],
            capture_output=True, text=True, timeout=300, cwd=str(BASE_DIR)
        )
        if result.returncode == 0:
            if (CACHE_DIR / "recent_matches.json").exists():
                with open(CACHE_DIR / "recent_matches.json") as f:
                    RECENT_MATCHES = json.load(f)
            if (CACHE_DIR / "active_players.json").exists():
                with open(CACHE_DIR / "active_players.json") as f:
                    ACTIVE_PLAYERS = json.load(f)
                ACTIVE_DICT = {p["player_id"]: p for p in ACTIVE_PLAYERS}
            REFRESH_STATE["last_status"] = "success"
            REFRESH_STATE["message"] = f"Done — {len(RECENT_MATCHES)} recent matches, {len(ACTIVE_PLAYERS)} active players"
        else:
            REFRESH_STATE["last_status"] = "error"
            REFRESH_STATE["message"] = (result.stderr or "Unknown error")[:300]
    except subprocess.TimeoutExpired:
        REFRESH_STATE["last_status"] = "error"
        REFRESH_STATE["message"] = "Refresh timed out after 5 minutes"
    except Exception as e:
        REFRESH_STATE["last_status"] = "error"
        REFRESH_STATE["message"] = str(e)
    finally:
        REFRESH_STATE["running"] = False
        REFRESH_STATE["last_refresh"] = datetime.now().isoformat()


@app.post("/api/refresh")
def trigger_refresh():
    if REFRESH_STATE["running"]:
        return {"status": "already_running", "message": "Refresh already in progress"}
    threading.Thread(target=_do_refresh, daemon=True).start()
    return {"status": "started", "message": "Data refresh started in background"}


@app.get("/api/refresh/status")
def refresh_status():
    latest = max((m["date"] for m in RECENT_MATCHES), default="N/A") if RECENT_MATCHES else "N/A"
    return {
        **REFRESH_STATE,
        "latest_match_date": latest,
        "total_recent_matches": len(RECENT_MATCHES),
        "total_active_players": len(ACTIVE_PLAYERS),
    }


# ============================================================
# ROUTES
# ============================================================
@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_META.get("model_type", "not loaded"),
        "model_loaded": MODEL is not None,
        "players": len(PLAYERS),
        "active_players": len(ACTIVE_PLAYERS),
        "recent_matches": len(RECENT_MATCHES),
        "paper_bets": len(PAPER_BETS),
        "startup_errors": _STARTUP_ERRORS,
    }


@app.get("/api/players")
def search_players(q: str = Query(..., min_length=2), limit: int = 20, active_only: bool = True):
    """Search players by name. Defaults to active players only."""
    q_lower = q.lower()

    if active_only and ACTIVE_PLAYERS:
        # Search active players (ranked, with recent data)
        results = [
            p for p in ACTIVE_PLAYERS
            if q_lower in p["name"].lower()
        ][:limit]
        return [
            {
                "player_id": p["player_id"],
                "name": p["name"],
                "country": p.get("country", ""),
                "hand": p.get("hand", ""),
                "tour": p.get("tour", ""),
                "rank": p.get("rank"),
                "elo": p.get("elo_overall"),
                "form_last_10": p.get("form_last_10"),
                "is_active": True,
            }
            for p in results
        ]

    # Fallback: search all players
    mask = PLAYERS["full_name"].str.lower().str.contains(q_lower, na=False)
    results = PLAYERS[mask].head(limit)
    return [
        {
            "player_id": int(r["player_id"]),
            "name": r["full_name"],
            "country": r.get("ioc", ""),
            "hand": r.get("hand", ""),
            "tour": r.get("tour", ""),
            "rank": ACTIVE_DICT.get(int(r["player_id"]), {}).get("rank"),
            "is_active": int(r["player_id"]) in ACTIVE_DICT,
        }
        for _, r in results.iterrows()
    ]


@app.get("/api/players/{player_id}")
def get_player(player_id: int):
    """Get full player profile."""
    info = get_player_info(player_id)
    if not info:
        raise HTTPException(status_code=404, detail="Player not found")

    # Recent matches from SQLite
    if MATCHES_DB.exists():
        conn = sqlite3.connect(MATCHES_DB)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM matches WHERE winner_id=? OR loser_id=? ORDER BY tourney_date DESC LIMIT 10",
            (player_id, player_id),
        )
        recent_rows = cur.fetchall()
        conn.close()
        info["recent_matches"] = [
            {
                "date": r["tourney_date"] or "",
                "tournament": r["tourney_name"] or "",
                "surface": r["surface"] or "",
                "round": r["round"] or "",
                "opponent": r["loser_name"] if int(r["winner_id"]) == player_id else r["winner_name"],
                "result": "W" if int(r["winner_id"]) == player_id else "L",
                "score": r["score"] or "",
            }
            for r in recent_rows
        ]

    return info


@app.get("/api/h2h/{p1_id}/{p2_id}")
def head_to_head(p1_id: int, p2_id: int):
    """Head-to-head record between two players."""
    p1 = get_player_info(p1_id)
    p2 = get_player_info(p2_id)
    if not p1 or not p2:
        raise HTTPException(status_code=404, detail="Player not found")

    h2h = get_h2h(p1_id, p2_id)
    h2h["player1"] = {"id": p1_id, "name": p1["name"]}
    h2h["player2"] = {"id": p2_id, "name": p2["name"]}
    return h2h


@app.post("/api/predict")
def predict_match(req: PredictRequest):
    """Predict match outcome between two players."""
    p1 = get_player_info(req.player1_id)
    p2 = get_player_info(req.player2_id)
    if not p1:
        raise HTTPException(status_code=404, detail=f"Player 1 ({req.player1_id}) not found")
    if not p2:
        raise HTTPException(status_code=404, detail=f"Player 2 ({req.player2_id}) not found")

    # Build features and predict
    X = build_features(req.player1_id, req.player2_id, req)
    prob_p1 = float(MODEL.predict(X, num_iteration=BEST_ITERATION)[0])
    prob_p2 = 1.0 - prob_p1

    # Value analysis
    value = {}
    if req.odds_p1 and req.odds_p1 > 1:
        impl_p1 = 1.0 / req.odds_p1
        edge_p1 = prob_p1 - impl_p1
        value["p1"] = {
            "odds": req.odds_p1,
            "implied_prob": round(impl_p1 * 100, 1),
            "model_prob": round(prob_p1 * 100, 1),
            "edge": round(edge_p1 * 100, 1),
            "is_value": edge_p1 > 0.05,
            "rating": "STRONG" if edge_p1 > 0.15 else "GOOD" if edge_p1 > 0.10 else "MILD" if edge_p1 > 0.05 else "NO VALUE",
        }
    if req.odds_p2 and req.odds_p2 > 1:
        impl_p2 = 1.0 / req.odds_p2
        edge_p2 = prob_p2 - impl_p2
        value["p2"] = {
            "odds": req.odds_p2,
            "implied_prob": round(impl_p2 * 100, 1),
            "model_prob": round(prob_p2 * 100, 1),
            "edge": round(edge_p2 * 100, 1),
            "is_value": edge_p2 > 0.05,
            "rating": "STRONG" if edge_p2 > 0.15 else "GOOD" if edge_p2 > 0.10 else "MILD" if edge_p2 > 0.05 else "NO VALUE",
        }

    # H2H
    h2h = get_h2h(req.player1_id, req.player2_id)

    # Kelly stake suggestion (if odds provided)
    kelly = {}
    if req.odds_p1 and req.odds_p1 > 1:
        b = req.odds_p1 - 1
        f = (b * prob_p1 - prob_p2) / b
        f = max(0, min(f, 0.10)) * 0.25  # Quarter Kelly, capped
        kelly["p1_quarter_kelly_pct"] = round(f * 100, 2)
    if req.odds_p2 and req.odds_p2 > 1:
        b = req.odds_p2 - 1
        f = (b * prob_p2 - prob_p1) / b
        f = max(0, min(f, 0.10)) * 0.25
        kelly["p2_quarter_kelly_pct"] = round(f * 100, 2)

    return {
        "player1": {"id": req.player1_id, "name": p1["name"], "elo": p1["elo"]},
        "player2": {"id": req.player2_id, "name": p2["name"], "elo": p2["elo"]},
        "prediction": {
            "p1_win_prob": round(prob_p1 * 100, 1),
            "p2_win_prob": round(prob_p2 * 100, 1),
            "predicted_winner": p1["name"] if prob_p1 > 0.5 else p2["name"],
            "confidence": round(max(prob_p1, prob_p2) * 100, 1),
        },
        "surface": req.surface,
        "round": req.round,
        "best_of": req.best_of,
        "h2h": {
            "p1_wins": h2h["p1_wins"],
            "p2_wins": h2h["p2_wins"],
        },
        "value": value,
        "kelly": kelly,
    }


# ============================================================
# PAPER BETTING
# ============================================================
@app.post("/api/paper-bets")
def create_paper_bet(req: PaperBetRequest):
    """Record a paper bet."""
    bet = {
        "id": len(PAPER_BETS) + 1,
        "created_at": datetime.now().isoformat(),
        "player1_id": req.player1_id,
        "player2_id": req.player2_id,
        "player1_name": req.player1_name,
        "player2_name": req.player2_name,
        "bet_on": req.bet_on,
        "bet_on_name": req.player1_name if req.bet_on == "p1" else req.player2_name,
        "odds": req.odds,
        "stake": req.stake,
        "model_prob": req.model_prob,
        "edge": req.edge,
        "surface": req.surface,
        "tournament": req.tournament,
        "notes": req.notes,
        "status": "pending",
        "won": None,
        "profit": None,
    }
    PAPER_BETS.append(bet)
    save_paper_bets()
    return bet


@app.get("/api/paper-bets")
def list_paper_bets(status: Optional[str] = None):
    """List paper bets, optionally filtered by status (pending/settled)."""
    bets = PAPER_BETS
    if status == "pending":
        bets = [b for b in bets if b["status"] == "pending"]
    elif status == "settled":
        bets = [b for b in bets if b["status"] == "settled"]
    return list(reversed(bets))  # Most recent first


@app.patch("/api/paper-bets/{bet_id}")
def settle_paper_bet(bet_id: int, req: SettleBetRequest):
    """Settle a paper bet — mark as won or lost."""
    bet = next((b for b in PAPER_BETS if b["id"] == bet_id), None)
    if not bet:
        raise HTTPException(status_code=404, detail="Bet not found")
    if bet["status"] == "settled":
        raise HTTPException(status_code=400, detail="Bet already settled")

    bet["won"] = req.won
    bet["status"] = "settled"
    bet["settled_at"] = datetime.now().isoformat()
    if req.won:
        bet["profit"] = round(bet["stake"] * (bet["odds"] - 1), 2)
    else:
        bet["profit"] = -bet["stake"]

    save_paper_bets()
    return bet


@app.get("/api/paper-bets/summary")
def paper_bets_summary():
    """Paper trading P&L summary."""
    settled = [b for b in PAPER_BETS if b["status"] == "settled"]
    pending = [b for b in PAPER_BETS if b["status"] == "pending"]

    if not settled:
        return {
            "total_bets": len(PAPER_BETS),
            "settled": 0,
            "pending": len(pending),
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "total_staked": 0,
            "total_profit": 0,
            "roi": 0,
            "pending_stake": sum(b["stake"] for b in pending),
        }

    wins = sum(1 for b in settled if b["won"])
    losses = len(settled) - wins
    total_staked = sum(b["stake"] for b in settled)
    total_profit = sum(b["profit"] for b in settled)
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0

    # Streak
    current_streak = 0
    for b in reversed(settled):
        if b["won"]:
            current_streak += 1
        else:
            break

    # Best/worst
    best_bet = max(settled, key=lambda b: b["profit"]) if settled else None
    worst_bet = min(settled, key=lambda b: b["profit"]) if settled else None

    return {
        "total_bets": len(PAPER_BETS),
        "settled": len(settled),
        "pending": len(pending),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / len(settled) * 100, 1),
        "total_staked": round(total_staked, 2),
        "total_profit": round(total_profit, 2),
        "roi": round(roi, 1),
        "current_win_streak": current_streak,
        "pending_stake": round(sum(b["stake"] for b in pending), 2),
        "avg_odds": round(np.mean([b["odds"] for b in settled]), 2),
        "avg_edge": round(np.mean([b["edge"] for b in settled]) * 100, 1),
        "best_bet": best_bet,
        "worst_bet": worst_bet,
    }


# ============================================================
# RECENT / LIVE MATCHES
# ============================================================
@app.get("/api/tournaments")
def get_tournaments():
    """Derive active/recent tournament list from recent match data."""
    from collections import defaultdict
    from datetime import datetime, timedelta

    tour_data: dict = defaultdict(lambda: {"matches": 0, "dates": [], "surface": "Hard", "level": ""})
    for m in RECENT_MATCHES:
        name = m.get("tournament") or m.get("tourney_name", "Unknown")
        if not name or name == "Unknown":
            continue
        d = tour_data[name]
        d["matches"] += 1
        if m.get("date"):
            d["dates"].append(m["date"])
        d["surface"] = m.get("surface", d["surface"])
        d["level"] = m.get("level", d["level"])

    now = datetime.now()
    cutoff_live = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    cutoff_upcoming = (now + timedelta(days=14)).strftime("%Y-%m-%d")

    results = []
    surface_images = {
        "Hard": "https://images.unsplash.com/photo-1595435934249-5df7ed86e1c0?auto=format&fit=crop&q=80&w=800",
        "Clay": "https://images.unsplash.com/photo-1554068865-24cecd4e34b8?auto=format&fit=crop&q=80&w=800",
        "Grass": "https://images.unsplash.com/photo-1599391398131-cd12dfc6c0ad?auto=format&fit=crop&q=80&w=800",
    }
    level_category = {
        "G": "Grand Slam", "M": "ATP Masters 1000", "F": "ATP Finals",
        "A": "ATP 500", "B": "ATP 250", "D": "Davis Cup",
    }

    for name, d in tour_data.items():
        dates = sorted(set(d["dates"]))
        if not dates:
            continue
        latest = dates[-1]
        earliest = dates[0]
        surface = d["surface"] or "Hard"
        status = "Finished"
        if latest >= cutoff_live:
            status = "Live"

        results.append({
            "id": name.lower().replace(" ", "-").replace("/", "-"),
            "name": name,
            "category": level_category.get(d["level"], "ATP Tour"),
            "surface": surface,
            "location": "",
            "date": f"{earliest} – {latest}",
            "status": status,
            "image": surface_images.get(surface, surface_images["Hard"]),
            "match_count": d["matches"],
        })

    results.sort(key=lambda x: (x["status"] != "Live", x["date"]), reverse=False)
    return results[:30]


@app.get("/api/matches/recent")
def recent_matches(limit: int = 50, tournament: Optional[str] = None):
    """Recent match results from ongoing tournaments."""
    matches = RECENT_MATCHES
    if tournament:
        matches = [m for m in matches if tournament.lower() in m.get("tournament", "").lower()]
    return matches[:limit]


@app.get("/api/stats/leaders")
def stats_leaders(category: str = "serve", tour: str = "atp", limit: int = 10):
    """Return top players ranked by a specific stat category using real data."""
    import math

    # Get active players for the tour
    players = [p for p in ACTIVE_PLAYERS if p.get("tour", "").lower() == tour.lower() and p.get("rank")]

    results = []
    for p in players:
        pid = p["player_id"]
        stats = STATS_DICT.get(pid, {})
        ap = ACTIVE_DICT.get(pid, {})

        def pct(v):
            """Convert 0-1 fraction to 0-100 percentage, or pass through if already 0-100."""
            if v is None or (isinstance(v, float) and math.isnan(v)): return None
            return round(v * 100, 1) if v <= 1.5 else round(v, 1)

        if category == "serve":
            val = stats.get("avg_1st_serve_pct", None)
            val2 = stats.get("avg_1st_serve_win_pct", None)
            val3 = stats.get("avg_ace_rate", None)
            pval = pct(val)
            if pval is None: continue
            results.append({
                "player_id": pid,
                "name": p["name"],
                "country": p.get("country", ""),
                "rank": p.get("rank"),
                "rank_points": p.get("rank_points", 0),
                "elo": p.get("elo_overall", 1500),
                "form_last_10": p.get("form_last_10", 0),
                "stat_primary": pval,
                "stat_primary_label": "1st Serve %",
                "stat_secondary": pct(val2) or 0,
                "stat_secondary_label": "1st Serve Win %",
                "stat_tertiary": pct(val3) or 0,
                "stat_tertiary_label": "Ace Rate",
                "sort_key": pval,
            })
        elif category == "return":
            val = ap.get("win_rate_hard", None) or stats.get("win_rate", None)
            if val is None or (isinstance(val, float) and math.isnan(val)): continue
            winr = stats.get("win_rate", 0) or 0
            results.append({
                "player_id": pid,
                "name": p["name"],
                "country": p.get("country", ""),
                "rank": p.get("rank"),
                "rank_points": p.get("rank_points", 0),
                "elo": p.get("elo_overall", 1500),
                "form_last_10": p.get("form_last_10", 0),
                "stat_primary": round(winr * 100, 1) if winr <= 1 else round(winr, 1),
                "stat_primary_label": "Overall Win %",
                "stat_secondary": round(ap.get("win_rate_hard", 0) * 100, 1) if ap.get("win_rate_hard", 0) <= 1 else round(ap.get("win_rate_hard", 0), 1),
                "stat_secondary_label": "Hard Win %",
                "stat_tertiary": round(ap.get("win_rate_clay", 0) * 100, 1) if ap.get("win_rate_clay", 0) <= 1 else round(ap.get("win_rate_clay", 0), 1),
                "stat_tertiary_label": "Clay Win %",
                "sort_key": winr,
            })
        elif category == "clutch":
            streak = ap.get("win_streak", 0) or 0
            form = p.get("form_last_10", 0) or 0
            if streak == 0 and form == 0: continue
            results.append({
                "player_id": pid,
                "name": p["name"],
                "country": p.get("country", ""),
                "rank": p.get("rank"),
                "rank_points": p.get("rank_points", 0),
                "elo": p.get("elo_overall", 1500),
                "form_last_10": form,
                "stat_primary": form,
                "stat_primary_label": "Form (Last 10, %)",
                "stat_secondary": streak,
                "stat_secondary_label": "Current Win Streak",
                "stat_tertiary": p.get("elo_overall", 1500),
                "stat_tertiary_label": "Elo Rating",
                "sort_key": form,
            })
        elif category == "physical":
            elo = p.get("elo_overall", 0) or 0
            if elo == 0: continue
            results.append({
                "player_id": pid,
                "name": p["name"],
                "country": p.get("country", ""),
                "rank": p.get("rank"),
                "rank_points": p.get("rank_points", 0),
                "elo": elo,
                "form_last_10": p.get("form_last_10", 0),
                "stat_primary": round(elo, 0),
                "stat_primary_label": "Elo Rating",
                "stat_secondary": round(ap.get("elo_hard", elo), 0),
                "stat_secondary_label": "Hard Court Elo",
                "stat_tertiary": round(ap.get("elo_clay", elo), 0),
                "stat_tertiary_label": "Clay Court Elo",
                "sort_key": elo,
            })

    results.sort(key=lambda x: x["sort_key"], reverse=True)
    return results[:limit]


@app.get("/api/rankings")
def rankings(tour: str = "atp", limit: int = 100):
    """Current rankings from active player data."""
    players = [
        p for p in ACTIVE_PLAYERS
        if p.get("rank") and p.get("tour", "").lower() == tour.lower()
    ]
    return [
        {
            "rank": p["rank"],
            "player_id": p["player_id"],
            "name": p["name"],
            "country": p.get("country", ""),
            "rank_points": p.get("rank_points", 0),
            "elo": p.get("elo_overall", 1500),
            "form_last_10": p.get("form_last_10", 0),
            "win_streak": p.get("win_streak", 0),
            "tour": p.get("tour", ""),
        }
        for p in players[:limit]
    ]


# ============================================================
# ODDS API
# ============================================================
class OddsConfigRequest(BaseModel):
    api_key: str


@app.post("/api/odds/config")
def configure_odds(req: OddsConfigRequest):
    """Set The Odds API key."""
    save_api_key(req.api_key)
    return {"status": "ok", "message": "API key saved"}


@app.get("/api/odds/config")
def get_odds_config():
    """Check if API key is configured."""
    key = load_api_key()
    return {"configured": bool(key), "key_preview": f"...{key[-6:]}" if key and len(key) > 6 else ""}


@app.get("/api/odds/live")
def get_live_odds(refresh: bool = False):
    """Get live tennis odds. Set refresh=true to fetch fresh from API."""
    if refresh:
        return fetch_live_odds()
    return get_cached_odds()


# ============================================================
# SCANNER / AUTO-BETTING
# ============================================================
SCAN_LOCK = threading.Lock()


@app.post("/api/scanner/run")
def trigger_scan(min_confidence: float = 0.55, min_edge: float = 0.03, auto_bet: bool = True):
    """Run the scanner to find value bets. Optionally auto-place paper bets."""
    if not SCAN_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Scan already in progress")
    try:
        results = run_scan(use_cached_odds=False, min_confidence=min_confidence, min_edge=min_edge)
        new_bets = []
        if auto_bet and results.get("auto_bet_candidates"):
            new_bets = auto_place_bets(results)
        results["auto_bets_placed"] = len(new_bets)
        return results
    finally:
        SCAN_LOCK.release()


@app.get("/api/scanner/results")
def get_scan_results():
    """Get the latest scan results."""
    scan_file = CACHE_DIR / "last_scan.json"
    if scan_file.exists():
        with open(scan_file) as f:
            return json.load(f)
    return {"matches_scanned": 0, "value_bets": [], "message": "No scan results yet. Run a scan first."}


@app.get("/api/auto-bets")
def get_auto_bets(status: Optional[str] = None):
    """Get auto-placed paper bets."""
    auto_file = CACHE_DIR / "auto_bets.json"
    if not auto_file.exists():
        return []
    with open(auto_file) as f:
        bets = json.load(f)
    if status == "pending":
        bets = [b for b in bets if b.get("status") == "pending"]
    elif status == "settled":
        bets = [b for b in bets if b.get("status") == "settled"]
    return list(reversed(bets))


@app.patch("/api/auto-bets/{bet_id}")
def settle_auto_bet(bet_id: int, req: SettleBetRequest):
    """Settle an auto-placed bet."""
    auto_file = CACHE_DIR / "auto_bets.json"
    if not auto_file.exists():
        raise HTTPException(status_code=404, detail="No auto bets file")
    with open(auto_file) as f:
        bets = json.load(f)

    bet = next((b for b in bets if b["id"] == bet_id), None)
    if not bet:
        raise HTTPException(status_code=404, detail="Bet not found")
    if bet.get("status") == "settled":
        raise HTTPException(status_code=400, detail="Already settled")

    bet["won"] = req.won
    bet["status"] = "settled"
    bet["settled_at"] = datetime.now().isoformat()
    bet["profit"] = round(bet["stake"] * (bet["odds"] - 1), 2) if req.won else -bet["stake"]

    with open(auto_file, "w") as f:
        json.dump(bets, f, indent=2, default=str)
    return bet


# ============================================================
# PERFORMANCE / SELF-EVALUATION
# ============================================================
@app.get("/api/performance")
def get_performance():
    """Get full performance evaluation with calibration and recommendations."""
    return evaluate_performance()


@app.get("/api/performance/summary")
def get_performance_summary():
    """Quick performance summary."""
    perf = evaluate_performance()
    return {
        "settled": perf.get("settled", 0),
        "win_rate": perf.get("win_rate", 0),
        "roi": perf.get("roi", 0),
        "total_profit": perf.get("total_profit", 0),
        "streak": perf.get("current_streak", "0"),
        "recommendations": perf.get("recommendations", []),
    }

# ============================================================
# WEBSOCKET: LIVE PULSE
# ============================================================
@app.get("/api/match-stats/{p1_id}/{p2_id}")
def match_comparison_stats(p1_id: int, p2_id: int, surface: str = "Hard"):
    """Career-average stat comparison for two players (used in MatchCenter)."""
    stats1 = STATS_DICT.get(p1_id, {})
    stats2 = STATS_DICT.get(p2_id, {})
    ap1 = ACTIVE_DICT.get(p1_id, {})
    ap2 = ACTIVE_DICT.get(p2_id, {})

    def pct(v, fallback: float = 0.0) -> float:
        if v is None:
            return fallback
        try:
            f = float(v)
            if np.isnan(f):
                return fallback
            # Values stored as 0-1 fraction → convert to percentage
            return round(f * 100, 1) if 0.0 <= f <= 1.0 else round(f, 1)
        except Exception:
            return fallback

    surf = surface.lower().replace("outdoor ", "").replace("indoor ", "")
    surf_key = f"win_rate_{surf}"
    elo_key = f"elo_{surf}"

    return {
        "service": [
            {"label": "1st Serve %",      "p1": pct(stats1.get("avg_1st_serve_pct"),     63.0), "p2": pct(stats2.get("avg_1st_serve_pct"),     63.0)},
            {"label": "1st Serve Win %",   "p1": pct(stats1.get("avg_1st_serve_win_pct"), 71.0), "p2": pct(stats2.get("avg_1st_serve_win_pct"), 71.0)},
            {"label": "Ace Rate",          "p1": pct(stats1.get("avg_ace_rate"),           5.0),  "p2": pct(stats2.get("avg_ace_rate"),           5.0)},
        ],
        "points": [
            {"label": "Overall Win Rate",  "p1": pct(ap1.get("win_rate") or stats1.get("win_rate"), 50.0), "p2": pct(ap2.get("win_rate") or stats2.get("win_rate"), 50.0)},
            {"label": f"{surface} Win %",  "p1": pct(stats1.get(surf_key) or ap1.get(surf_key),    50.0), "p2": pct(stats2.get(surf_key) or ap2.get(surf_key),    50.0)},
            {"label": "Form (Last 10)",    "p1": pct(ap1.get("form_last_10") or stats1.get("form_last_10"), 50.0), "p2": pct(ap2.get("form_last_10") or stats2.get("form_last_10"), 50.0)},
        ],
        "games": [
            {"label": "Current Win Streak", "p1": int(ap1.get("win_streak", 0)), "p2": int(ap2.get("win_streak", 0))},
            {"label": "Matches (6 months)", "p1": int(ap1.get("matches_6m", 0)), "p2": int(ap2.get("matches_6m", 0))},
            {"label": "World Rank",         "p1": ap1.get("rank") or 999,        "p2": ap2.get("rank") or 999, "lower_is_better": True},
        ],
        "intelligence": [
            {"label": "Elo Rating",         "p1": round(float(ap1.get("elo_overall", 1500)), 0), "p2": round(float(ap2.get("elo_overall", 1500)), 0)},
            {"label": f"{surface} Elo",     "p1": round(float(ap1.get(elo_key, 1500)), 0),       "p2": round(float(ap2.get(elo_key, 1500)), 0)},
        ],
    }


@app.websocket("/ws/match/{match_id}/pulse")
async def websocket_match_pulse(websocket: WebSocket, match_id: str):
    await pulse_manager.connect(websocket, match_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Can process incoming client messages here if needed
    except WebSocketDisconnect:
        pulse_manager.disconnect(websocket, match_id)


# ============================================================
# NEWS
# ============================================================
@app.get("/api/news")
def news_feed(q: Optional[str] = None, limit: int = 60, days: int = 5):
    """Latest tennis news (past 5 days by default, stored in SQLite)."""
    return get_cached_news(q=q, limit=limit, days=days)


@app.get("/api/news/live")
def live_match_news(p1: str = "", p2: str = "", limit: int = 20):
    """
    News relevant to a live match — player mentions + incident keywords.
    Frontend polls this every 5 min while a live match is open.
    """
    if not p1 and not p2:
        raise HTTPException(status_code=400, detail="Provide p1= and/or p2= player names")
    return get_live_match_news(p1, p2, limit=limit)


@app.post("/api/news/refresh")
def refresh_news_feed():
    """Manually trigger a fresh fetch from all RSS feeds."""
    try:
        result = fetch_news()
        return {"status": "ok", "item_count": result.get("item_count", 0), "failures": result.get("failures", [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

