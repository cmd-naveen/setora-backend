"""
Tennis News Fetcher — SQLite-backed, full content, no external links.

- Fetches ATP Tour, Tennis.com, ESPN Tennis RSS feeds
- Extracts full article text (no images)
- Stores in SQLite (cache/news.db)
- Auto-deletes articles older than 7 days
- Returns only past-5-days articles by default
- Live match news: filters for player names + incident keywords
"""
from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional

import feedparser
import requests

log = logging.getLogger("news")

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
DB_PATH = CACHE_DIR / "news.db"

FEEDS: List[Dict[str, Any]] = [
    {"name": "ATP Tour",    "url": "https://www.atptour.com/en/media/rss-feed/xml-feed"},
    {"name": "Tennis.com",  "url": "https://www.tennis.com/roots/rss-feeds/news/"},
    {"name": "ESPN Tennis", "url": "https://www.espn.com/espn/rss/tennis/news"},
]

LIVE_KEYWORDS = [
    "injur", "retire", "retired hurt", "delay", "postpone", "cancel",
    "withdraw", "walkover", "default", "medical", "suspended", "rain",
    "darkness", "ankle", "wrist", "back", "knee", "illness", "sick",
]

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 15
KEEP_DAYS = 7
SHOW_DAYS = 5


# ── SQLite setup ───────────────────────────────────────────────────────────────

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id          TEXT PRIMARY KEY,
            source      TEXT,
            title       TEXT NOT NULL,
            url         TEXT UNIQUE,
            content     TEXT,
            summary     TEXT,
            category    TEXT,
            published_at TEXT,
            fetched_at  TEXT,
            author      TEXT
        )
    """)
    conn.commit()
    return conn


# ── HTML content extractor ─────────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    """Strip HTML tags; skip script/style/nav/header/footer."""
    _SKIP = {"script", "style", "nav", "header", "footer", "aside",
              "noscript", "iframe", "figure", "figcaption", "button",
              "form", "input", "select", "option", "meta", "link"}
    _BLOCK = {"p", "div", "article", "section", "h1", "h2", "h3",
               "h4", "h5", "h6", "li", "blockquote", "td", "tr"}

    def __init__(self):
        super().__init__()
        self._depth = 0          # nesting depth of skipped tags
        self._parts: list[str] = []
        self._pending_newline = False

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._depth += 1
        elif tag in self._BLOCK and self._depth == 0:
            self._pending_newline = True

    def handle_endtag(self, tag):
        if tag in self._SKIP:
            self._depth = max(0, self._depth - 1)
        elif tag in self._BLOCK and self._depth == 0:
            self._pending_newline = True

    def handle_data(self, data):
        if self._depth > 0:
            return
        text = data.strip()
        if not text:
            return
        if self._pending_newline and self._parts:
            self._parts.append("\n")
            self._pending_newline = False
        self._parts.append(text)

    def get_text(self) -> str:
        raw = " ".join(self._parts)
        # Collapse multiple spaces/newlines
        raw = re.sub(r" {2,}", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _fetch_article_content(url: str) -> str:
    """Download URL and extract article text. Returns empty string on failure."""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        resp.raise_for_status()
        parser = _TextExtractor()
        parser.feed(resp.text)
        text = parser.get_text()
        # Keep reasonable length — 8000 chars max
        return text[:8000] if text else ""
    except Exception as e:
        log.debug(f"Content fetch failed for {url}: {e}")
        return ""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _uid(link: str) -> str:
    return hashlib.md5(link.encode()).hexdigest()[:16]


def _parse_dt(entry: dict) -> Optional[datetime]:
    for attr in ("published_parsed", "updated_parsed"):
        t = entry.get(attr)
        if t:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    for field in ("published", "updated", "pubDate"):
        val = entry.get(field)
        if val:
            try:
                return parsedate_to_datetime(str(val)).astimezone(timezone.utc)
            except Exception:
                pass
    return None


def _time_ago(iso: Optional[str]) -> str:
    if not iso:
        return "Recently"
    try:
        dt = datetime.fromisoformat(iso)
        s = int((datetime.now(timezone.utc) - dt).total_seconds())
        if s < 3600:
            m = max(1, s // 60)
            return f"{m} min{'s' if m != 1 else ''} ago"
        if s < 86400:
            h = s // 3600
            return f"{h} hr{'s' if h != 1 else ''} ago"
        if s < 172800:
            return "Yesterday"
        return f"{s // 86400} days ago"
    except Exception:
        return "Recently"


def _clean(text: Optional[str]) -> str:
    if not text:
        return ""
    # Strip basic HTML tags from RSS summaries
    text = re.sub(r"<[^>]+>", " ", str(text))
    return " ".join(text.split())


# ── Cleanup ────────────────────────────────────────────────────────────────────

def cleanup_old_articles():
    """Delete articles older than KEEP_DAYS days."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=KEEP_DAYS)).isoformat()
    with _db() as conn:
        cur = conn.execute(
            "DELETE FROM articles WHERE published_at < ? AND published_at IS NOT NULL",
            (cutoff,)
        )
        deleted = cur.rowcount
        if deleted:
            log.info(f"Cleaned up {deleted} articles older than {KEEP_DAYS} days")


# ── Feed fetch + store ─────────────────────────────────────────────────────────

def _fetch_and_store_feed(name: str, url: str) -> int:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.8",
    }
    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    parsed = feedparser.parse(resp.content)
    if parsed.bozo and not parsed.entries:
        raise RuntimeError(str(getattr(parsed, "bozo_exception", "parse error")))

    now_iso = datetime.now(timezone.utc).isoformat()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=KEEP_DAYS)).isoformat()
    stored = 0

    with _db() as conn:
        existing_ids = {row[0] for row in conn.execute("SELECT id FROM articles")}

        for entry in parsed.entries:
            title = _clean(entry.get("title"))
            link = entry.get("link", "")
            if not title or not link:
                continue

            dt = _parse_dt(entry)
            if dt and dt.isoformat() < cutoff:
                continue  # too old

            uid = _uid(link)
            if uid in existing_ids:
                continue  # already stored

            # Get content: try RSS summary first, then fetch URL
            rss_summary = _clean(entry.get("summary") or entry.get("description") or "")

            if len(rss_summary) >= 300:
                content = rss_summary
            else:
                # Fetch full article content
                content = _fetch_article_content(link)
                if not content:
                    content = rss_summary
                time.sleep(0.2)  # be polite

            summary = content[:400].rsplit(" ", 1)[0] + "…" if len(content) > 400 else content

            conn.execute("""
                INSERT OR IGNORE INTO articles
                    (id, source, title, url, content, summary, category, published_at, fetched_at, author)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                uid,
                name,
                title,
                link,
                content,
                summary,
                name,
                dt.isoformat() if dt else now_iso,
                now_iso,
                _clean(entry.get("author")) or None,
            ))
            stored += 1

    return stored


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_news() -> dict:
    """Fetch all feeds, store new articles, cleanup old ones."""
    total = 0
    failures = []

    for feed in FEEDS:
        try:
            n = _fetch_and_store_feed(feed["name"], feed["url"])
            log.info(f"  {feed['name']}: +{n} new articles")
            total += n
        except Exception as e:
            log.warning(f"  {feed['name']} FAILED: {e}")
            failures.append({"name": feed["name"], "error": str(e)})
        time.sleep(0.5)

    cleanup_old_articles()

    return {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "item_count": total,
        "failures": failures,
    }


def get_cached_news(q: Optional[str] = None, limit: int = 60, days: int = SHOW_DAYS) -> dict:
    """Return stored articles from past `days` days. Optional keyword filter."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    with _db() as conn:
        if q:
            ql = f"%{q.lower()}%"
            rows = conn.execute("""
                SELECT * FROM articles
                WHERE published_at >= ?
                  AND (LOWER(title) LIKE ? OR LOWER(content) LIKE ?)
                ORDER BY published_at DESC
                LIMIT ?
            """, (cutoff, ql, ql, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM articles
                WHERE published_at >= ?
                ORDER BY published_at DESC
                LIMIT ?
            """, (cutoff, limit)).fetchall()

    items = []
    for row in rows:
        d = dict(row)
        d["time"] = _time_ago(d.get("published_at"))
        items.append(d)

    return {
        "items": items,
        "item_count": len(items),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def get_live_match_news(player1: str, player2: str, limit: int = 20) -> dict:
    """
    Return recent news relevant to a live match.
    Searches by player names AND match-incident keywords.
    Past 2 days only for freshness.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    p1_parts = player1.lower().split()
    p2_parts = player2.lower().split()

    # Search terms: last names are most distinctive
    terms = []
    for parts in (p1_parts, p2_parts):
        if parts:
            terms.append(parts[-1])  # last name
            if len(parts) > 1:
                terms.append(parts[0])  # first name too

    with _db() as conn:
        # Get all recent articles and filter in Python for flexibility
        rows = conn.execute("""
            SELECT * FROM articles
            WHERE published_at >= ?
            ORDER BY published_at DESC
            LIMIT 200
        """, (cutoff,)).fetchall()

    items = []
    for row in rows:
        d = dict(row)
        text = f"{d.get('title', '')} {d.get('content', '')}".lower()

        # Must mention at least one player
        mentions_player = any(term in text for term in terms)
        if not mentions_player:
            continue

        # Tag as live-relevant if it mentions incident keywords
        is_incident = any(kw in text for kw in LIVE_KEYWORDS)
        d["is_incident"] = is_incident
        d["time"] = _time_ago(d.get("published_at"))
        items.append(d)

        if len(items) >= limit:
            break

    # Sort: incidents first, then by date
    items.sort(key=lambda x: (not x["is_incident"], x.get("published_at", "")), reverse=True)

    return {
        "items": items[:limit],
        "item_count": len(items),
        "players": [player1, player2],
    }
