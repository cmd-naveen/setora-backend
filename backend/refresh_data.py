"""
Refresh live data: downloads latest matches, computes active player index,
builds name→ID lookup for cross-source matching.

Usage:
  python refresh_data.py
"""

import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "data-pipeline" / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

TML_BASE = "https://stats.tennismylife.org/data"
SACKMANN_WTA_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master"

print("=== Refreshing live data ===")

# ── 1. Download fresh data ──
print("Downloading ongoing tournaments...")
try:
    r = requests.get(f"{TML_BASE}/ongoing_tourneys.csv", timeout=30)
    if r.status_code == 200:
        (RAW_DIR / "tennismylife" / "ongoing_tourneys.csv").write_bytes(r.content)
        print(f"  ongoing_tourneys.csv updated ({len(r.content)} bytes)")
except Exception as e:
    print(f"  Failed: {e}")

# Download fresh 2025/2026 ATP data from TML
for fname in ["2025.csv", "2026.csv", "2026_challenger.csv"]:
    print(f"Downloading TML {fname}...")
    try:
        r = requests.get(f"{TML_BASE}/{fname}", timeout=30)
        if r.status_code == 200 and len(r.content) > 100:
            (RAW_DIR / "tennismylife" / fname).write_bytes(r.content)
            print(f"  {fname} updated ({len(r.content)} bytes)")
        else:
            print(f"  {fname}: not available or empty")
    except Exception as e:
        print(f"  Failed: {e}")

# Download fresh WTA 2025 data from Sackmann (fills the WTA gap)
print("Downloading Sackmann WTA 2025 data...")
wta_dir = RAW_DIR / "wta"
wta_dir.mkdir(parents=True, exist_ok=True)
for fname in ["wta_matches_2025.csv", "wta_matches_qual_itf_2025.csv"]:
    try:
        r = requests.get(f"{SACKMANN_WTA_BASE}/{fname}", timeout=30)
        if r.status_code == 200 and len(r.content) > 100:
            (wta_dir / fname).write_bytes(r.content)
            print(f"  {fname} updated ({len(r.content)} bytes)")
        else:
            print(f"  {fname}: not available yet")
    except Exception as e:
        print(f"  Failed: {e}")

# Also try WTA 2026
for fname in ["wta_matches_2026.csv"]:
    try:
        r = requests.get(f"{SACKMANN_WTA_BASE}/{fname}", timeout=30)
        if r.status_code == 200 and len(r.content) > 100:
            (wta_dir / fname).write_bytes(r.content)
            print(f"  {fname} updated ({len(r.content)} bytes)")
        else:
            print(f"  {fname}: not available yet")
    except Exception as e:
        print(f"  Failed: {e}")

# ── 2. Build active players index ──
print("\nBuilding active players index...")

# Load the main processed matches
matches = pd.read_csv(PROCESSED_DIR / "matches_clean.csv", low_memory=False,
    usecols=["winner_id", "loser_id", "winner_name", "loser_name",
             "tourney_date", "tourney_name", "surface", "round",
             "winner_rank", "loser_rank", "winner_rank_points", "loser_rank_points",
             "score", "tour"])
matches["tourney_date"] = pd.to_datetime(matches["tourney_date"], errors="coerce")
matches = matches.dropna(subset=["tourney_date"])

# Also load fresh WTA 2025/2026 data directly (may not be in matches_clean yet)
wta_fresh_frames = []
for fname in ["wta_matches_2025.csv", "wta_matches_qual_itf_2025.csv", "wta_matches_2026.csv"]:
    p = wta_dir / fname
    if p.exists():
        try:
            df = pd.read_csv(p, low_memory=False)
            # Normalize column names to match our format
            if "tourney_date" in df.columns:
                df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
            if "tourney_id" in df.columns and "tour" not in df.columns:
                df["tour"] = "wta"
            wta_fresh_frames.append(df)
            print(f"  Loaded fresh WTA: {fname} ({len(df)} matches)")
        except Exception as e:
            print(f"  Error loading {fname}: {e}")

if wta_fresh_frames:
    wta_fresh = pd.concat(wta_fresh_frames, ignore_index=True)
    # Ensure required columns exist
    for col in ["winner_id", "loser_id", "winner_name", "loser_name", "tourney_date",
                "tourney_name", "surface", "round", "winner_rank", "loser_rank",
                "winner_rank_points", "loser_rank_points", "score", "tour"]:
        if col not in wta_fresh.columns:
            wta_fresh[col] = np.nan if col.endswith("rank") or col.endswith("points") else ""
    wta_fresh = wta_fresh.dropna(subset=["tourney_date"])
    matches = pd.concat([matches, wta_fresh[matches.columns]], ignore_index=True)
    print(f"  Total matches after adding fresh WTA: {len(matches)}")

# Load TML ongoing tournaments
ongoing = pd.DataFrame()
ongoing_path = RAW_DIR / "tennismylife" / "ongoing_tourneys.csv"
if ongoing_path.exists():
    ongoing = pd.read_csv(ongoing_path, low_memory=False)
    if "tourney_date" in ongoing.columns:
        ongoing["tourney_date"] = pd.to_datetime(ongoing["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")

# Load TML 2025/2026 data for additional ATP matches
tml_frames = []
for fname in ["2025.csv", "2026.csv", "2026_challenger.csv"]:
    p = RAW_DIR / "tennismylife" / fname
    if p.exists():
        try:
            df = pd.read_csv(p, low_memory=False)
            if "tourney_date" in df.columns:
                df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
            if "tour" not in df.columns:
                df["tour"] = "atp"
            tml_frames.append(df)
            print(f"  Loaded TML: {fname} ({len(df)} matches)")
        except Exception as e:
            print(f"  Error loading {fname}: {e}")

# Load players
players = pd.read_csv(PROCESSED_DIR / "players.csv", low_memory=False)
players["full_name"] = (players["name_first"].fillna("") + " " + players["name_last"].fillna("")).str.strip()

# Build name→player_id lookup for cross-source matching
name_to_id = {}
for _, p in players.iterrows():
    name = p["full_name"].lower().strip()
    pid = int(p["player_id"])
    name_to_id[name] = pid
    # Also index by "Last, First" and "Last First" variations
    parts = p["full_name"].split()
    if len(parts) >= 2:
        # "Sinner Jannik" → also map reversed
        rev = " ".join(parts[::-1]).lower()
        if rev not in name_to_id:
            name_to_id[rev] = pid

# Merge TML data into matches (TML uses different ID formats, resolve by name)
if tml_frames:
    tml_all = pd.concat(tml_frames, ignore_index=True)
    tml_all = tml_all.dropna(subset=["tourney_date"])

    # Resolve TML player names to Sackmann IDs
    def resolve_id(name, tml_id):
        if pd.isna(name):
            return np.nan
        n = str(name).lower().strip()
        if n in name_to_id:
            return name_to_id[n]
        # Try partial match
        for key, pid in name_to_id.items():
            if n in key or key in n:
                return pid
        return np.nan

    tml_all["winner_id_resolved"] = tml_all.apply(
        lambda r: resolve_id(r.get("winner_name"), r.get("winner_id")), axis=1)
    tml_all["loser_id_resolved"] = tml_all.apply(
        lambda r: resolve_id(r.get("loser_name"), r.get("loser_id")), axis=1)

    # Use resolved IDs
    tml_all["winner_id"] = tml_all["winner_id_resolved"]
    tml_all["loser_id"] = tml_all["loser_id_resolved"]

    # Ensure columns
    for col in ["winner_rank_points", "loser_rank_points", "tour"]:
        if col not in tml_all.columns:
            tml_all[col] = "" if col == "tour" else np.nan
    if "tour" in tml_all.columns:
        tml_all["tour"] = tml_all["tour"].fillna("atp")

    # Only add rows with resolved IDs
    resolved = tml_all.dropna(subset=["winner_id", "loser_id"])
    if not resolved.empty:
        for col in matches.columns:
            if col not in resolved.columns:
                resolved[col] = np.nan
        matches = pd.concat([matches, resolved[matches.columns]], ignore_index=True)
        print(f"  Added {len(resolved)} TML matches with resolved IDs")

# Remove duplicates (same players, same date, same tournament)
before = len(matches)
matches = matches.drop_duplicates(
    subset=["winner_id", "loser_id", "tourney_date", "tourney_name"], keep="last")
print(f"  Deduped: {before} → {len(matches)} matches")

# Load Elo ratings
elo = pd.read_csv(PROCESSED_DIR / "elo_ratings.csv", low_memory=False)
elo_dict = elo.set_index("player_id").to_dict("index")

# Load player stats
stats = pd.read_csv(PROCESSED_DIR / "player_stats.csv", low_memory=False)
stats_dict = stats.set_index("player_id").to_dict("index")

# Track per-player: last match, last rank, recent results
player_data = {}
cutoff_active = datetime.now() - timedelta(days=550)
cutoff_recent = datetime.now() - timedelta(days=365)

# Process all matches sorted by date
sorted_m = matches.sort_values("tourney_date")
for _, row in sorted_m.iterrows():
    d = row["tourney_date"]
    for pid_col, name_col, rank_col, pts_col, is_winner in [
        ("winner_id", "winner_name", "winner_rank", "winner_rank_points", True),
        ("loser_id", "loser_name", "loser_rank", "loser_rank_points", False),
    ]:
        pid = row.get(pid_col)
        if pd.isna(pid):
            continue
        pid = int(pid)
        name = str(row.get(name_col, ""))
        rank = row.get(rank_col)
        pts = row.get(pts_col)
        tour = row.get("tour", "")

        if pid not in player_data:
            player_data[pid] = {
                "name": name,
                "tour": tour,
                "last_match": d,
                "last_rank": None,
                "last_rank_pts": None,
                "recent_results": [],
            }

        pd_entry = player_data[pid]
        if d >= pd_entry["last_match"]:
            pd_entry["last_match"] = d
            pd_entry["name"] = name
            if tour:
                pd_entry["tour"] = tour

        if pd.notna(rank) and (pd_entry["last_rank"] is None or d >= pd_entry["last_match"]):
            pd_entry["last_rank"] = int(rank)
            pd_entry["last_rank_pts"] = int(pts) if pd.notna(pts) else 0

        if d >= cutoff_recent:
            opponent = row.get("loser_name" if is_winner else "winner_name", "")
            pd_entry["recent_results"].append({
                "date": d.strftime("%Y-%m-%d"),
                "won": is_winner,
                "surface": row.get("surface", ""),
                "tourney": row.get("tourney_name", ""),
                "opponent": str(opponent),
                "score": str(row.get("score", "")),
                "round": str(row.get("round", "")),
            })

# Build the active players JSON
active_players = []
for pid, pd_entry in player_data.items():
    is_active = pd_entry["last_match"] >= cutoff_active
    if not is_active:
        continue

    p_row = players[players["player_id"] == pid]
    hand = ""
    height = None
    country = ""
    dob = None
    if not p_row.empty:
        p = p_row.iloc[0]
        hand = str(p.get("hand", ""))
        height = int(p["height"]) if pd.notna(p.get("height")) else None
        country = str(p.get("ioc", ""))
        dob = str(p.get("dob", "")) if pd.notna(p.get("dob")) else None

    elo_data = elo_dict.get(pid, {})
    st = stats_dict.get(pid, {})

    recent = pd_entry["recent_results"]
    recent_wins = sum(1 for r in recent if r["won"])
    recent_form = recent_wins / len(recent) if recent else 0

    last_10 = recent[-10:]
    form_10 = sum(1 for r in last_10 if r["won"]) / len(last_10) if last_10 else 0

    streak = 0
    for r in reversed(recent):
        if r["won"]:
            streak += 1
        else:
            break

    days_since = (datetime.now() - pd_entry["last_match"]).days

    active_players.append({
        "player_id": pid,
        "name": pd_entry["name"],
        "tour": pd_entry["tour"],
        "hand": hand,
        "height": height,
        "country": country,
        "dob": dob,
        "rank": pd_entry["last_rank"],
        "rank_points": pd_entry["last_rank_pts"],
        "elo_overall": round(elo_data.get("elo_overall", 1500), 1),
        "elo_hard": round(elo_data.get("elo_hard", 1500), 1),
        "elo_clay": round(elo_data.get("elo_clay", 1500), 1),
        "elo_grass": round(elo_data.get("elo_grass", 1500), 1),
        "total_matches": int(st.get("total_matches", 0)),
        "win_rate": round(st.get("win_rate", 0) * 100, 1),
        "win_rate_hard": round(st.get("win_rate_hard", 0) * 100, 1),
        "win_rate_clay": round(st.get("win_rate_clay", 0) * 100, 1),
        "win_rate_grass": round(st.get("win_rate_grass", 0) * 100, 1),
        "form_6m": round(recent_form * 100, 1),
        "form_last_10": round(form_10 * 100, 1),
        "win_streak": streak,
        "days_since_match": days_since,
        "matches_6m": len(recent),
        "last_match_date": pd_entry["last_match"].strftime("%Y-%m-%d"),
        "recent_matches": recent[-10:][::-1],
    })

active_players.sort(key=lambda p: (p["rank"] or 9999, -p["elo_overall"]))

print(f"  Active players: {len(active_players)}")
print(f"  With ranking: {sum(1 for p in active_players if p['rank'])}")
print(f"  ATP: {sum(1 for p in active_players if p['tour'] == 'atp')}")
print(f"  WTA: {sum(1 for p in active_players if p['tour'] == 'wta')}")

# Check WTA form data
wta_with_form = sum(1 for p in active_players if p['tour'] == 'wta' and p['form_last_10'] > 0)
print(f"  WTA with form data: {wta_with_form}")

with open(CACHE_DIR / "active_players.json", "w") as f:
    json.dump(active_players, f)
print(f"  Saved active_players.json")

# ── 3. Build name→ID mapping for cross-source lookups ──
print("\nBuilding name→ID lookup...")
name_lookup = {}
for p in active_players:
    name_lookup[p["name"].lower()] = p["player_id"]
    # Also add without accents/common variations
    parts = p["name"].split()
    if len(parts) >= 2:
        # "Last, First" format
        name_lookup[f"{parts[-1]} {' '.join(parts[:-1])}".lower()] = p["player_id"]

with open(CACHE_DIR / "name_to_id.json", "w") as f:
    json.dump(name_lookup, f)
print(f"  Name lookup entries: {len(name_lookup)}")

# ── 4. Build recent matches with resolved IDs ──
print("\nBuilding recent matches feed...")

recent_matches = []
if not ongoing.empty:
    for _, row in ongoing.iterrows():
        w_name = str(row.get("winner_name", ""))
        l_name = str(row.get("loser_name", ""))
        # Resolve to Sackmann player_id via name lookup
        w_id = name_lookup.get(w_name.lower(), None)
        l_id = name_lookup.get(l_name.lower(), None)

        recent_matches.append({
            "date": row["tourney_date"].strftime("%Y-%m-%d") if pd.notna(row.get("tourney_date")) else "",
            "tournament": str(row.get("tourney_name", "")),
            "surface": str(row.get("surface", "")),
            "round": str(row.get("round", "")),
            "level": str(row.get("tourney_level", "")),
            "winner_id": w_id,
            "winner_name": w_name,
            "winner_rank": int(row["winner_rank"]) if pd.notna(row.get("winner_rank")) else None,
            "loser_id": l_id,
            "loser_name": l_name,
            "loser_rank": int(row["loser_rank"]) if pd.notna(row.get("loser_rank")) else None,
            "score": str(row.get("score", "")),
        })

with open(CACHE_DIR / "recent_matches.json", "w") as f:
    json.dump(recent_matches, f)
print(f"  Recent matches: {len(recent_matches)}")

# Top 10 sample
for tour_name in ["atp", "wta"]:
    top = [p for p in active_players if p["rank"] and p["tour"] == tour_name][:10]
    if top:
        print(f"\nTop 10 {tour_name.upper()}:")
        for p in top:
            print(f"  #{p['rank']} {p['name']} (Elo: {p['elo_overall']}, Form L10: {p['form_last_10']}%, Streak: {p['win_streak']}W, Matches 6M: {p['matches_6m']})")

print("\n=== Refresh complete ===")
