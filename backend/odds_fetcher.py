"""
Odds Fetcher — Retrieves live tennis odds from The Odds API.

Free tier: 500 requests/month. Get your key at https://the-odds-api.com

The API uses tournament-specific sport keys (e.g. tennis_atp_miami_open),
not generic tennis_atp_singles. We dynamically discover active tennis tournaments.
"""

import json
import os
import requests
from pathlib import Path
from datetime import datetime, timedelta

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
ODDS_CACHE = CACHE_DIR / "live_odds.json"
ODDS_CONFIG = CACHE_DIR / "odds_config.json"

ODDS_API_BASE = "https://api.the-odds-api.com/v4"


def load_api_key():
    # Prefer env var (Railway/production), fall back to local config file
    env_key = os.environ.get("ODDS_API_KEY", "")
    if env_key:
        return env_key
    if ODDS_CONFIG.exists():
        with open(ODDS_CONFIG) as f:
            return json.load(f).get("api_key", "")
    return ""


def save_api_key(key):
    config = {}
    if ODDS_CONFIG.exists():
        with open(ODDS_CONFIG) as f:
            config = json.load(f)
    config["api_key"] = key
    with open(ODDS_CONFIG, "w") as f:
        json.dump(config, f, indent=2)


def get_active_tennis_sports(api_key):
    """Discover all currently active tennis tournaments on the API."""
    try:
        resp = requests.get(
            f"{ODDS_API_BASE}/sports",
            params={"apiKey": api_key},
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"  Sports list error: {resp.status_code}")
            return []
        sports = resp.json()
        tennis = [
            s for s in sports
            if s.get("key", "").startswith("tennis_") and s.get("active", False)
        ]
        print(f"  Active tennis tournaments: {len(tennis)}")
        for s in tennis:
            print(f"    {s['key']} — {s['title']}")
        return tennis
    except Exception as e:
        print(f"  Error fetching sports: {e}")
        return []


def fetch_live_odds(api_key=None):
    """
    Fetch live tennis odds from The Odds API.
    Dynamically discovers active tennis tournaments, then fetches odds for each.
    """
    api_key = api_key or load_api_key()
    if not api_key:
        return {"error": "No API key configured. Set it via /api/odds/config", "matches": []}

    # Step 1: Discover active tennis tournaments
    tennis_sports = get_active_tennis_sports(api_key)
    if not tennis_sports:
        return {"error": "No active tennis tournaments found", "matches": []}

    all_matches = []
    requests_used = 0
    remaining = "?"

    for sport_info in tennis_sports:
        sport_key = sport_info["key"]
        sport_title = sport_info.get("title", "")

        # Determine tour from key
        if "_atp_" in sport_key:
            tour = "atp"
        elif "_wta_" in sport_key:
            tour = "wta"
        else:
            tour = "atp"

        try:
            resp = requests.get(
                f"{ODDS_API_BASE}/sports/{sport_key}/odds",
                params={
                    "apiKey": api_key,
                    "regions": "eu,uk",
                    "markets": "h2h",
                    "oddsFormat": "decimal",
                },
                timeout=15,
            )

            requests_used += 1
            remaining = resp.headers.get("x-requests-remaining", remaining)

            if resp.status_code == 401:
                return {"error": "Invalid API key", "matches": all_matches}
            if resp.status_code == 429:
                return {"error": "Rate limit exceeded. Free tier: 500 req/month", "matches": all_matches,
                        "requests_used": requests_used, "remaining_requests": remaining}
            if resp.status_code != 200:
                print(f"  {sport_key}: status {resp.status_code}")
                continue

            data = resp.json()
            print(f"  {sport_key}: {len(data)} events")

            for event in data:
                match = {
                    "id": event.get("id", ""),
                    "sport": sport_key,
                    "sport_title": sport_title,
                    "tour": tour,
                    "home": event.get("home_team", ""),
                    "away": event.get("away_team", ""),
                    "commence_time": event.get("commence_time", ""),
                    "bookmakers": [],
                    "best_odds": {},
                }

                best_home = 0
                best_away = 0

                for bk in event.get("bookmakers", []):
                    bk_name = bk.get("title", "")
                    for market in bk.get("markets", []):
                        if market.get("key") == "h2h":
                            outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                            home_odds = outcomes.get(event.get("home_team"), 0)
                            away_odds = outcomes.get(event.get("away_team"), 0)

                            if home_odds > 0 and away_odds > 0:
                                match["bookmakers"].append({
                                    "name": bk_name,
                                    "home_odds": home_odds,
                                    "away_odds": away_odds,
                                    "last_update": bk.get("last_update", ""),
                                })

                                if home_odds > best_home:
                                    best_home = home_odds
                                if away_odds > best_away:
                                    best_away = away_odds

                if match["bookmakers"]:
                    avg_home = sum(b["home_odds"] for b in match["bookmakers"]) / len(match["bookmakers"])
                    avg_away = sum(b["away_odds"] for b in match["bookmakers"]) / len(match["bookmakers"])

                    match["best_odds"] = {
                        "home": round(best_home, 2),
                        "away": round(best_away, 2),
                        "avg_home": round(avg_home, 2),
                        "avg_away": round(avg_away, 2),
                    }
                    all_matches.append(match)

        except Exception as e:
            print(f"  Error fetching {sport_key}: {e}")

    # Cache results
    result = {
        "fetched_at": datetime.now().isoformat(),
        "requests_used": requests_used,
        "remaining_requests": remaining,
        "tournaments": [s["title"] for s in tennis_sports],
        "matches": all_matches,
    }
    with open(ODDS_CACHE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Total matches with odds: {len(all_matches)}")
    print(f"  Requests used this call: {requests_used}")
    print(f"  Remaining this month: {remaining}")

    return result


def get_cached_odds():
    """Get cached odds (avoids hitting API repeatedly)."""
    if ODDS_CACHE.exists():
        with open(ODDS_CACHE) as f:
            data = json.load(f)
        fetched = data.get("fetched_at", "")
        if fetched:
            try:
                age = datetime.now() - datetime.fromisoformat(fetched)
                data["age_minutes"] = round(age.total_seconds() / 60, 1)
                data["is_stale"] = age > timedelta(hours=1)
            except Exception:
                data["is_stale"] = True
        return data
    return {"matches": [], "is_stale": True, "error": "No cached odds. Fetch first."}


def normalize_player_name(name):
    """Normalize player name for matching."""
    return name.lower().strip().replace("-", " ").replace("'", "")


def match_odds_to_players(odds_matches, name_to_id, active_dict):
    """
    Match odds data to our player database.
    Returns enriched matches with player IDs and model-ready data.
    """
    matched = []
    for m in odds_matches:
        home_name = normalize_player_name(m.get("home", ""))
        away_name = normalize_player_name(m.get("away", ""))

        home_id = name_to_id.get(home_name)
        away_id = name_to_id.get(away_name)

        # Try partial matching if exact fails
        if not home_id:
            for key, pid in name_to_id.items():
                if home_name in key or key in home_name:
                    home_id = pid
                    break
        if not away_id:
            for key, pid in name_to_id.items():
                if away_name in key or key in away_name:
                    away_id = pid
                    break

        m["home_player_id"] = home_id
        m["away_player_id"] = away_id
        m["home_matched"] = home_id is not None
        m["away_matched"] = away_id is not None
        m["both_matched"] = home_id is not None and away_id is not None

        if home_id and home_id in active_dict:
            ap = active_dict[home_id]
            m["home_rank"] = ap.get("rank")
            m["home_elo"] = ap.get("elo_overall")
        if away_id and away_id in active_dict:
            ap = active_dict[away_id]
            m["away_rank"] = ap.get("rank")
            m["away_elo"] = ap.get("elo_overall")

        matched.append(m)

    return matched
