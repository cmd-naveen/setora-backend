const API = "http://localhost:8000";

async function fetchJSON(url, opts = {}) {
  const res = await fetch(API + url, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export const searchPlayers = (q) => fetchJSON(`/api/players?q=${encodeURIComponent(q)}`);
export const getPlayer = (id) => fetchJSON(`/api/players/${id}`);
export const getH2H = (p1, p2) => fetchJSON(`/api/h2h/${p1}/${p2}`);

export const predict = (data) =>
  fetchJSON("/api/predict", { method: "POST", body: JSON.stringify(data) });

export const getPaperBets = (status) =>
  fetchJSON(`/api/paper-bets${status ? `?status=${status}` : ""}`);

export const createPaperBet = (data) =>
  fetchJSON("/api/paper-bets", { method: "POST", body: JSON.stringify(data) });

export const settleBet = (id, won) =>
  fetchJSON(`/api/paper-bets/${id}`, {
    method: "PATCH",
    body: JSON.stringify({ won }),
  });

export const getPaperBetsSummary = () => fetchJSON("/api/paper-bets/summary");

export const getRecentMatches = (limit = 50) => fetchJSON(`/api/matches/recent?limit=${limit}`);
export const getRankings = (tour = "atp", limit = 100) =>
  fetchJSON(`/api/rankings?tour=${tour}&limit=${limit}`);

// Odds API
export const getOddsConfig = () => fetchJSON("/api/odds/config");
export const setOddsApiKey = (key) =>
  fetchJSON("/api/odds/config", { method: "POST", body: JSON.stringify({ api_key: key }) });
export const getLiveOdds = (refresh = false) =>
  fetchJSON(`/api/odds/live?refresh=${refresh}`);

// Scanner / Auto-betting
export const runScanner = (autoBet = true) =>
  fetchJSON(`/api/scanner/run?auto_bet=${autoBet}`, { method: "POST" });
export const getScanResults = () => fetchJSON("/api/scanner/results");
export const getAutoBets = (status) =>
  fetchJSON(`/api/auto-bets${status ? `?status=${status}` : ""}`);
export const settleAutoBet = (id, won) =>
  fetchJSON(`/api/auto-bets/${id}`, {
    method: "PATCH",
    body: JSON.stringify({ won }),
  });

// Performance
export const getPerformance = () => fetchJSON("/api/performance");
export const getPerformanceSummary = () => fetchJSON("/api/performance/summary");
