import { useState, useEffect, useCallback } from "react";
import {
  searchPlayers,
  getPlayer,
  predict,
  createPaperBet,
  getPaperBets,
  settleBet,
  getPaperBetsSummary,
  getRecentMatches,
  getRankings,
  getOddsConfig,
  setOddsApiKey,
  getLiveOdds,
  runScanner,
  getScanResults,
  getAutoBets,
  settleAutoBet,
  getPerformance,
} from "./api";
import "./App.css";

// ── Player Search Input ──
function PlayerSearch({ label, onSelect, selected }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (query.length < 2) { setResults([]); return; }
    const t = setTimeout(async () => {
      try {
        const r = await searchPlayers(query);
        setResults(r);
        setOpen(true);
      } catch { setResults([]); }
    }, 250);
    return () => clearTimeout(t);
  }, [query]);

  return (
    <div className="search-box">
      <label style={{ fontSize: 12, color: "var(--text2)", marginBottom: 4, display: "block" }}>
        {label}
      </label>
      {selected ? (
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontWeight: 600 }}>{selected.name}</span>
          {selected.rank && (
            <span className="rank-badge">#{selected.rank}</span>
          )}
          {selected.country && (
            <span style={{ fontSize: 12, color: "var(--text2)" }}>{selected.country}</span>
          )}
          <button
            style={{ background: "var(--bg3)", fontSize: 12, padding: "2px 8px" }}
            onClick={() => { onSelect(null); setQuery(""); }}
          >
            change
          </button>
        </div>
      ) : (
        <input
          placeholder="Search player..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => results.length && setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 200)}
        />
      )}
      {open && results.length > 0 && (
        <div className="search-results">
          {results.map((p) => (
            <div
              key={p.player_id}
              onClick={() => { onSelect(p); setOpen(false); setQuery(""); }}
            >
              <span>
                {p.rank && <span className="rank-badge" style={{ marginRight: 6 }}>#{p.rank}</span>}
                {p.name}
                {p.form_last_10 != null && (
                  <span style={{ fontSize: 11, color: "var(--text2)", marginLeft: 6 }}>
                    Form: {p.form_last_10}%
                  </span>
                )}
              </span>
              <span className="country">
                {p.country} / {(p.tour || "").toUpperCase()}
                {p.is_active === false && <span className="retired-badge">RETIRED</span>}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Predict Page ──
function PredictPage() {
  const [p1, setP1] = useState(null);
  const [p2, setP2] = useState(null);
  const [surface, setSurface] = useState("Hard");
  const [level, setLevel] = useState("A");
  const [round, setRound] = useState("R32");
  const [bestOf, setBestOf] = useState(3);
  const [oddsP1, setOddsP1] = useState("");
  const [oddsP2, setOddsP2] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [betMsg, setBetMsg] = useState("");

  const handlePredict = async () => {
    if (!p1 || !p2) return;
    setLoading(true);
    setBetMsg("");
    try {
      const r = await predict({
        player1_id: p1.player_id,
        player2_id: p2.player_id,
        surface,
        tourney_level: level,
        round,
        best_of: bestOf,
        tour: p1.tour || "atp",
        odds_p1: oddsP1 ? parseFloat(oddsP1) : null,
        odds_p2: oddsP2 ? parseFloat(oddsP2) : null,
      });
      setResult(r);
    } catch (e) {
      alert("Prediction failed: " + e.message);
    }
    setLoading(false);
  };

  const handlePaperBet = async (betOn) => {
    if (!result) return;
    const v = result.value[betOn];
    if (!v) { setBetMsg("No odds provided for this player"); return; }
    try {
      await createPaperBet({
        player1_id: result.player1.id,
        player2_id: result.player2.id,
        player1_name: result.player1.name,
        player2_name: result.player2.name,
        bet_on: betOn,
        odds: v.odds,
        stake: 100,
        model_prob: v.model_prob / 100,
        edge: v.edge / 100,
        surface,
        tournament: "",
      });
      setBetMsg(`Paper bet placed on ${betOn === "p1" ? result.player1.name : result.player2.name} @ ${v.odds}`);
    } catch (e) {
      setBetMsg("Failed: " + e.message);
    }
  };

  return (
    <div>
      <div className="predict-grid">
        <div>
          <PlayerSearch label="Player 1" selected={p1} onSelect={setP1} />
          {p1 && (
            <div className="player-mini-card">
              {p1.rank && <span>Rank #{p1.rank}</span>}
              {p1.elo && <span>Elo {p1.elo}</span>}
              {p1.form_last_10 != null && (
                <span className={p1.form_last_10 >= 60 ? "form-hot" : p1.form_last_10 >= 40 ? "form-ok" : "form-cold"}>
                  Form {p1.form_last_10}%
                </span>
              )}
            </div>
          )}
          <input
            placeholder="Odds (e.g. 1.72)"
            value={oddsP1}
            onChange={(e) => setOddsP1(e.target.value)}
            style={{ marginTop: 8 }}
          />
        </div>
        <div className="vs">VS</div>
        <div>
          <PlayerSearch label="Player 2" selected={p2} onSelect={setP2} />
          {p2 && (
            <div className="player-mini-card">
              {p2.rank && <span>Rank #{p2.rank}</span>}
              {p2.elo && <span>Elo {p2.elo}</span>}
              {p2.form_last_10 != null && (
                <span className={p2.form_last_10 >= 60 ? "form-hot" : p2.form_last_10 >= 40 ? "form-ok" : "form-cold"}>
                  Form {p2.form_last_10}%
                </span>
              )}
            </div>
          )}
          <input
            placeholder="Odds (e.g. 2.15)"
            value={oddsP2}
            onChange={(e) => setOddsP2(e.target.value)}
            style={{ marginTop: 8 }}
          />
        </div>
      </div>

      <div className="match-options">
        <div>
          <label>Surface</label>
          <select value={surface} onChange={(e) => setSurface(e.target.value)}>
            <option>Hard</option>
            <option>Clay</option>
            <option>Grass</option>
          </select>
        </div>
        <div>
          <label>Level</label>
          <select value={level} onChange={(e) => setLevel(e.target.value)}>
            <option value="G">Grand Slam</option>
            <option value="M">Masters</option>
            <option value="A">ATP/WTA</option>
            <option value="C">Challenger</option>
          </select>
        </div>
        <div>
          <label>Round</label>
          <select value={round} onChange={(e) => setRound(e.target.value)}>
            <option value="F">Final</option>
            <option value="SF">Semi-Final</option>
            <option value="QF">Quarter-Final</option>
            <option value="R16">Round of 16</option>
            <option value="R32">Round of 32</option>
            <option value="R64">Round of 64</option>
            <option value="R128">Round of 128</option>
          </select>
        </div>
        <div>
          <label>Best of</label>
          <select value={bestOf} onChange={(e) => setBestOf(Number(e.target.value))}>
            <option value={3}>3 sets</option>
            <option value={5}>5 sets</option>
          </select>
        </div>
      </div>

      <button className="predict-btn" onClick={handlePredict} disabled={!p1 || !p2 || loading}>
        {loading ? "Predicting..." : "Predict Match"}
      </button>

      {result && (
        <div className="result-card card">
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
            <span style={{ fontWeight: 600 }}>{result.player1.name}</span>
            <span style={{ fontWeight: 600 }}>{result.player2.name}</span>
          </div>

          <div className="prob-bar">
            <div className="p1" style={{ width: `${result.prediction.p1_win_prob}%` }}>
              {result.prediction.p1_win_prob}%
            </div>
            <div className="p2" style={{ width: `${result.prediction.p2_win_prob}%` }}>
              {result.prediction.p2_win_prob}%
            </div>
          </div>

          <div className="h2h-row">
            <span>H2H: <strong>{result.h2h.p1_wins}</strong> - <strong>{result.h2h.p2_wins}</strong></span>
            <span>Predicted: <strong>{result.prediction.predicted_winner}</strong></span>
            <span>Confidence: <strong>{result.prediction.confidence}%</strong></span>
          </div>

          {/* Elo comparison */}
          {result.player1.elo && result.player2.elo && (
            <div className="elo-compare">
              <div>
                <span className="label">Elo</span>
                <span className="val">{result.player1.elo.overall}</span>
              </div>
              <div style={{ color: "var(--text2)", fontSize: 12 }}>vs</div>
              <div>
                <span className="label">Elo</span>
                <span className="val">{result.player2.elo.overall}</span>
              </div>
            </div>
          )}

          {(result.value.p1 || result.value.p2) && (
            <div className="value-grid">
              {["p1", "p2"].map((pk) => {
                const v = result.value[pk];
                if (!v) return null;
                const name = pk === "p1" ? result.player1.name : result.player2.name;
                const badgeClass =
                  v.rating === "STRONG" ? "badge-strong" :
                  v.rating === "GOOD" ? "badge-good" :
                  v.rating === "MILD" ? "badge-mild" : "badge-none";
                return (
                  <div className="value-box" key={pk}>
                    <div className="label">{name}</div>
                    <div className="val">{v.model_prob}%</div>
                    <div style={{ fontSize: 13, color: "var(--text2)" }}>
                      Market: {v.implied_prob}% | Edge: {v.edge > 0 ? "+" : ""}{v.edge}%
                    </div>
                    <span className={`badge ${badgeClass}`}>{v.rating}</span>
                    {v.is_value && (
                      <button
                        className="btn-win"
                        style={{ marginTop: 8, display: "block", width: "100%" }}
                        onClick={() => handlePaperBet(pk)}
                      >
                        Paper Bet @ {v.odds}
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {result.kelly && Object.keys(result.kelly).length > 0 && (
            <div className="kelly-box">
              Kelly Stake:{" "}
              {result.kelly.p1_quarter_kelly_pct > 0 && (
                <span>{result.player1.name}: <strong>{result.kelly.p1_quarter_kelly_pct}%</strong> of bankroll</span>
              )}
              {result.kelly.p1_quarter_kelly_pct > 0 && result.kelly.p2_quarter_kelly_pct > 0 && " | "}
              {result.kelly.p2_quarter_kelly_pct > 0 && (
                <span>{result.player2.name}: <strong>{result.kelly.p2_quarter_kelly_pct}%</strong> of bankroll</span>
              )}
            </div>
          )}

          {betMsg && (
            <div style={{ marginTop: 8, fontSize: 13, color: "var(--green)" }}>{betMsg}</div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Paper Bets Page ──
function PaperBetsPage() {
  const [bets, setBets] = useState([]);
  const [summary, setSummary] = useState(null);
  const [filter, setFilter] = useState("");

  const load = useCallback(async () => {
    try {
      const [b, s] = await Promise.all([getPaperBets(filter), getPaperBetsSummary()]);
      setBets(b);
      setSummary(s);
    } catch { /* ignore */ }
  }, [filter]);

  useEffect(() => { load(); }, [load]);

  const handleSettle = async (id, won) => {
    await settleBet(id, won);
    load();
  };

  return (
    <div>
      {summary && summary.settled > 0 && (
        <div className="bets-summary">
          <div className="stat-box">
            <div className="label">Total Bets</div>
            <div className="val">{summary.total_bets}</div>
          </div>
          <div className="stat-box">
            <div className="label">Win Rate</div>
            <div className="val" style={{ color: summary.win_rate >= 50 ? "var(--green)" : "var(--red)" }}>
              {summary.win_rate}%
            </div>
          </div>
          <div className="stat-box">
            <div className="label">Profit</div>
            <div className="val" style={{ color: summary.total_profit >= 0 ? "var(--green)" : "var(--red)" }}>
              ${summary.total_profit?.toFixed(0)}
            </div>
          </div>
          <div className="stat-box">
            <div className="label">ROI</div>
            <div className="val" style={{ color: summary.roi >= 0 ? "var(--green)" : "var(--red)" }}>
              {summary.roi > 0 ? "+" : ""}{summary.roi}%
            </div>
          </div>
          <div className="stat-box">
            <div className="label">Pending</div>
            <div className="val" style={{ color: "var(--yellow)" }}>{summary.pending}</div>
          </div>
          <div className="stat-box">
            <div className="label">Avg Odds</div>
            <div className="val">{summary.avg_odds || "-"}</div>
          </div>
        </div>
      )}

      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        {["", "pending", "settled"].map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            style={{
              background: filter === f ? "var(--accent)" : "var(--bg3)",
              fontSize: 13,
            }}
          >
            {f === "" ? "All" : f.charAt(0).toUpperCase() + f.slice(1)}
          </button>
        ))}
      </div>

      <div className="bet-list">
        {bets.length === 0 && (
          <div style={{ color: "var(--text2)", textAlign: "center", padding: 40 }}>
            No paper bets yet. Go to Predict to place your first!
          </div>
        )}
        {bets.map((b) => (
          <div className="bet-item" key={b.id}>
            <div>
              <div className="matchup">
                {b.player1_name} vs {b.player2_name}
              </div>
              <div className="bet-details">
                Bet on: <strong>{b.bet_on_name}</strong> @ {b.odds} | Stake: ${b.stake} |
                Edge: {(b.edge * 100).toFixed(1)}%
                {b.surface && ` | ${b.surface}`}
              </div>
            </div>
            <div style={{ textAlign: "right" }}>
              {b.status === "pending" ? (
                <div className="bet-actions">
                  <button className="btn-win" onClick={() => handleSettle(b.id, true)}>Won</button>
                  <button className="btn-loss" onClick={() => handleSettle(b.id, false)}>Lost</button>
                </div>
              ) : (
                <div>
                  <span className={b.profit >= 0 ? "profit-pos" : "profit-neg"}>
                    {b.profit >= 0 ? "+" : ""}${b.profit?.toFixed(0)}
                  </span>
                  <div style={{ fontSize: 12, color: "var(--text2)" }}>
                    {b.won ? "WON" : "LOST"}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Player Profile Page ──
function PlayerPage({ initialPlayer, onClear }) {
  const [search, setSearch] = useState(null);
  const [player, setPlayer] = useState(null);

  // Handle navigation from other pages
  useEffect(() => {
    if (initialPlayer && initialPlayer.player_id) {
      setSearch(initialPlayer);
      if (onClear) onClear();
    }
  }, [initialPlayer, onClear]);

  useEffect(() => {
    if (!search) { setPlayer(null); return; }
    getPlayer(search.player_id).then(setPlayer).catch(() => {});
  }, [search]);

  return (
    <div>
      <PlayerSearch label="Search Player" selected={search} onSelect={setSearch} />
      {player && (
        <div className="card" style={{ marginTop: 16 }}>
          <div className="profile-header">
            <div>
              <div className="big-name">
                {player.name}
                {player.is_active ? (
                  <span className="active-badge">ACTIVE</span>
                ) : (
                  <span className="retired-badge">RETIRED</span>
                )}
              </div>
              <div className="details">
                {player.country} | {player.hand === "R" ? "Right-handed" : player.hand === "L" ? "Left-handed" : "Unknown"}
                {player.height && ` | ${player.height}cm`}
                {player.tour && ` | ${player.tour.toUpperCase()}`}
                {player.rank && ` | Rank #${player.rank}`}
                {player.rank_points && ` (${player.rank_points} pts)`}
              </div>
              {player.is_active && player.stats && (
                <div className="player-form-row">
                  {player.stats.win_streak > 0 && (
                    <span className="streak-badge">{player.stats.win_streak}W streak</span>
                  )}
                  {player.stats.form_last_10 != null && (
                    <span className={`form-badge ${player.stats.form_last_10 >= 60 ? "form-hot" : player.stats.form_last_10 >= 40 ? "form-ok" : "form-cold"}`}>
                      Form L10: {player.stats.form_last_10}%
                    </span>
                  )}
                  {player.days_since_match != null && (
                    <span style={{ fontSize: 12, color: "var(--text2)" }}>
                      Last match: {player.days_since_match}d ago
                    </span>
                  )}
                  {player.last_match_date && (
                    <span style={{ fontSize: 12, color: "var(--text2)" }}>
                      ({player.last_match_date})
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>

          <h3 style={{ fontSize: 14, color: "var(--text2)", marginBottom: 8 }}>Elo Ratings</h3>
          <div className="stats-grid">
            {Object.entries(player.elo).map(([k, v]) => (
              <div className="stat-box" key={k}>
                <div className="label">{k}</div>
                <div className="val" style={{ fontSize: 18 }}>{v}</div>
              </div>
            ))}
          </div>

          <h3 style={{ fontSize: 14, color: "var(--text2)", margin: "16px 0 8px" }}>Stats</h3>
          <div className="stats-grid">
            <div className="stat-box">
              <div className="label">Matches</div>
              <div className="val" style={{ fontSize: 18 }}>{player.stats.total_matches}</div>
            </div>
            <div className="stat-box">
              <div className="label">Win Rate</div>
              <div className="val" style={{ fontSize: 18 }}>{player.stats.win_rate}%</div>
            </div>
            {player.stats.form_last_10 != null && (
              <div className="stat-box">
                <div className="label">Form (L10)</div>
                <div className="val" style={{ fontSize: 18 }}>{player.stats.form_last_10}%</div>
              </div>
            )}
            {player.stats.win_rate_hard != null && (
              <div className="stat-box">
                <div className="label">Hard</div>
                <div className="val" style={{ fontSize: 18 }}>{player.stats.win_rate_hard}%</div>
              </div>
            )}
            {player.stats.win_rate_clay != null && (
              <div className="stat-box">
                <div className="label">Clay</div>
                <div className="val" style={{ fontSize: 18 }}>{player.stats.win_rate_clay}%</div>
              </div>
            )}
            {player.stats.win_rate_grass != null && (
              <div className="stat-box">
                <div className="label">Grass</div>
                <div className="val" style={{ fontSize: 18 }}>{player.stats.win_rate_grass}%</div>
              </div>
            )}
            {player.stats.form_6m != null && (
              <div className="stat-box">
                <div className="label">Form 6M</div>
                <div className="val" style={{ fontSize: 18 }}>{player.stats.form_6m}%</div>
              </div>
            )}
            {player.stats.matches_6m != null && (
              <div className="stat-box">
                <div className="label">Matches 6M</div>
                <div className="val" style={{ fontSize: 18 }}>{player.stats.matches_6m}</div>
              </div>
            )}
          </div>

          {player.recent_matches?.length > 0 && (
            <>
              <h3 style={{ fontSize: 14, color: "var(--text2)", margin: "16px 0 8px" }}>Recent Matches</h3>
              <table className="recent-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Tournament</th>
                    <th>Rd</th>
                    <th>Opponent</th>
                    <th>Result</th>
                    <th>Score</th>
                  </tr>
                </thead>
                <tbody>
                  {player.recent_matches.map((m, i) => (
                    <tr key={i}>
                      <td>{m.date}</td>
                      <td>{m.tournament || m.tourney}</td>
                      <td>{m.round}</td>
                      <td>{m.opponent}</td>
                      <td className={m.result === "W" || m.won ? "result-w" : "result-l"}>
                        {m.result || (m.won ? "W" : "L")}
                      </td>
                      <td>{m.score}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ── Live / Recent Matches Page ──
function LivePage({ onPlayerClick }) {
  const [matches, setMatches] = useState([]);
  const [rankings, setRankings] = useState([]);
  const [tab, setTab] = useState("matches");
  const [tour, setTour] = useState("atp");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    if (tab === "matches") {
      getRecentMatches(50).then(setMatches).catch(() => setMatches([])).finally(() => setLoading(false));
    } else {
      getRankings(tour, 100).then(setRankings).catch(() => setRankings([])).finally(() => setLoading(false));
    }
  }, [tab, tour]);

  // Group matches by tournament
  const tournaments = {};
  matches.forEach((m) => {
    const key = m.tournament || "Unknown";
    if (!tournaments[key]) tournaments[key] = { surface: m.surface, level: m.level, matches: [] };
    tournaments[key].matches.push(m);
  });

  return (
    <div>
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <button
          onClick={() => setTab("matches")}
          style={{ background: tab === "matches" ? "var(--accent)" : "var(--bg3)", fontSize: 13 }}
        >
          Recent Matches
        </button>
        <button
          onClick={() => setTab("rankings")}
          style={{ background: tab === "rankings" ? "var(--accent)" : "var(--bg3)", fontSize: 13 }}
        >
          Rankings
        </button>
      </div>

      {loading && <div style={{ color: "var(--text2)", textAlign: "center", padding: 20 }}>Loading...</div>}

      {!loading && tab === "matches" && (
        <>
          {matches.length === 0 ? (
            <div style={{ color: "var(--text2)", textAlign: "center", padding: 40 }}>
              No recent match data available. Run refresh_data.py to update.
            </div>
          ) : (
            Object.entries(tournaments).map(([name, t]) => (
              <div key={name} className="card" style={{ marginBottom: 16 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
                  <h3 style={{ fontSize: 16, fontWeight: 600 }}>{name}</h3>
                  <span style={{ fontSize: 12, color: "var(--text2)" }}>
                    {t.surface} {t.level && `| ${t.level}`}
                  </span>
                </div>
                <table className="recent-table">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Rd</th>
                      <th>Winner</th>
                      <th></th>
                      <th>Loser</th>
                      <th>Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {t.matches.map((m, i) => (
                      <tr key={i}>
                        <td>{m.date}</td>
                        <td>{m.round}</td>
                        <td>
                          <span
                            className={`result-w ${m.winner_id ? "player-link" : ""}`}
                            onClick={() => m.winner_id && onPlayerClick?.(m.winner_id, m.winner_name)}
                          >
                            {m.winner_name}
                          </span>
                          {m.winner_rank && <span style={{ fontSize: 11, color: "var(--text2)" }}> #{m.winner_rank}</span>}
                        </td>
                        <td style={{ color: "var(--text2)", fontSize: 11 }}>def.</td>
                        <td>
                          <span
                            className={m.loser_id ? "player-link" : ""}
                            onClick={() => m.loser_id && onPlayerClick?.(m.loser_id, m.loser_name)}
                          >
                            {m.loser_name}
                          </span>
                          {m.loser_rank && <span style={{ fontSize: 11, color: "var(--text2)" }}> #{m.loser_rank}</span>}
                        </td>
                        <td>{m.score}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ))
          )}
        </>
      )}

      {!loading && tab === "rankings" && (
        <div className="card">
          <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
            <button
              onClick={() => setTour("atp")}
              style={{ background: tour === "atp" ? "var(--accent)" : "var(--bg3)", fontSize: 13 }}
            >
              ATP
            </button>
            <button
              onClick={() => setTour("wta")}
              style={{ background: tour === "wta" ? "var(--accent)" : "var(--bg3)", fontSize: 13 }}
            >
              WTA
            </button>
          </div>
          {rankings.length === 0 ? (
            <div style={{ color: "var(--text2)", textAlign: "center", padding: 40 }}>
              No ranking data available.
            </div>
          ) : (
            <table className="recent-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Player</th>
                  <th>Country</th>
                  <th>Points</th>
                  <th>Elo</th>
                  <th>Form L10</th>
                  <th>Streak</th>
                </tr>
              </thead>
              <tbody>
                {rankings.map((p) => (
                  <tr key={p.player_id}>
                    <td style={{ fontWeight: 600 }}>#{p.rank}</td>
                    <td>{p.name}</td>
                    <td>{p.country}</td>
                    <td>{p.rank_points}</td>
                    <td>{p.elo}</td>
                    <td>
                      <span className={p.form_last_10 >= 60 ? "form-hot" : p.form_last_10 >= 40 ? "form-ok" : "form-cold"}>
                        {p.form_last_10}%
                      </span>
                    </td>
                    <td>{p.win_streak > 0 ? `${p.win_streak}W` : "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}

// ── Dashboard Page (Scanner + Auto-Bets + Performance) ──
function DashboardPage() {
  const [tab, setTab] = useState("scanner");
  const [scanResults, setScanResults] = useState(null);
  const [autoBets, setAutoBets] = useState([]);
  const [performance, setPerformance] = useState(null);
  const [oddsConfigured, setOddsConfigured] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [scanning, setScanning] = useState(false);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [config, scan, bets, perf] = await Promise.all([
        getOddsConfig().catch(() => ({ configured: false })),
        getScanResults().catch(() => null),
        getAutoBets().catch(() => []),
        getPerformance().catch(() => null),
      ]);
      setOddsConfigured(config.configured);
      setScanResults(scan);
      setAutoBets(bets);
      setPerformance(perf);
    } catch { /* ignore */ }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleSaveKey = async () => {
    if (!apiKey.trim()) return;
    try {
      await setOddsApiKey(apiKey.trim());
      setOddsConfigured(true);
      setApiKey("");
    } catch (e) {
      alert("Failed: " + e.message);
    }
  };

  const handleScan = async () => {
    setScanning(true);
    try {
      const results = await runScanner(true);
      setScanResults(results);
      // Reload auto bets and performance
      const [bets, perf] = await Promise.all([getAutoBets(), getPerformance()]);
      setAutoBets(bets);
      setPerformance(perf);
    } catch (e) {
      alert("Scan failed: " + e.message);
    }
    setScanning(false);
  };

  const handleSettleAuto = async (id, won) => {
    try {
      await settleAutoBet(id, won);
      const [bets, perf] = await Promise.all([getAutoBets(), getPerformance()]);
      setAutoBets(bets);
      setPerformance(perf);
    } catch (e) {
      alert("Failed: " + e.message);
    }
  };

  if (loading) return <div style={{ color: "var(--text2)", textAlign: "center", padding: 40 }}>Loading dashboard...</div>;

  return (
    <div>
      {/* API Key Setup */}
      {!oddsConfigured && (
        <div className="card" style={{ marginBottom: 16, borderLeft: "3px solid var(--yellow)" }}>
          <h3 style={{ fontSize: 14, marginBottom: 8 }}>Setup: Connect Live Odds</h3>
          <p style={{ fontSize: 13, color: "var(--text2)", marginBottom: 8 }}>
            Get a free API key from the-odds-api.com (500 req/month free) to enable live odds scanning.
          </p>
          <div style={{ display: "flex", gap: 8 }}>
            <input
              placeholder="Paste your API key..."
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              style={{ flex: 1 }}
            />
            <button onClick={handleSaveKey} className="predict-btn" style={{ width: "auto", padding: "8px 16px" }}>
              Save Key
            </button>
          </div>
        </div>
      )}

      {/* Performance Summary Bar */}
      {performance && performance.settled > 0 && (
        <div className="bets-summary" style={{ marginBottom: 16 }}>
          <div className="stat-box">
            <div className="label">Win Rate</div>
            <div className="val" style={{ color: performance.win_rate >= 55 ? "var(--green)" : "var(--red)" }}>
              {performance.win_rate}%
            </div>
          </div>
          <div className="stat-box">
            <div className="label">ROI</div>
            <div className="val" style={{ color: performance.roi >= 0 ? "var(--green)" : "var(--red)" }}>
              {performance.roi > 0 ? "+" : ""}{performance.roi}%
            </div>
          </div>
          <div className="stat-box">
            <div className="label">Profit</div>
            <div className="val" style={{ color: performance.total_profit >= 0 ? "var(--green)" : "var(--red)" }}>
              ${performance.total_profit}
            </div>
          </div>
          <div className="stat-box">
            <div className="label">Settled</div>
            <div className="val">{performance.settled}</div>
          </div>
          <div className="stat-box">
            <div className="label">Streak</div>
            <div className="val">{performance.current_streak}</div>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        {["scanner", "auto-bets", "performance"].map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{ background: tab === t ? "var(--accent)" : "var(--bg3)", fontSize: 13 }}
          >
            {t === "scanner" ? "Scanner" : t === "auto-bets" ? "Auto Bets" : "Performance"}
          </button>
        ))}
        <div style={{ flex: 1 }} />
        <button
          onClick={handleScan}
          disabled={scanning || !oddsConfigured}
          className="predict-btn"
          style={{ width: "auto", padding: "8px 20px", fontSize: 13, opacity: !oddsConfigured ? 0.5 : 1 }}
        >
          {scanning ? "Scanning..." : "Run Scanner"}
        </button>
      </div>

      {/* Scanner Tab */}
      {tab === "scanner" && (
        <div>
          {!scanResults || !scanResults.value_bets?.length ? (
            <div style={{ color: "var(--text2)", textAlign: "center", padding: 40 }}>
              {oddsConfigured
                ? 'No scan results yet. Click "Run Scanner" to find value bets.'
                : "Set up your Odds API key above to start scanning."}
            </div>
          ) : (
            <>
              <div style={{ fontSize: 13, color: "var(--text2)", marginBottom: 12 }}>
                Scanned {scanResults.matches_scanned} matches |
                Found {scanResults.value_bets?.length || 0} value bets |
                {scanResults.auto_bet_candidates?.length || 0} auto-bet candidates |
                Last scan: {scanResults.scan_time ? new Date(scanResults.scan_time).toLocaleString() : "never"}
              </div>

              {/* High confidence alerts */}
              {scanResults.auto_bet_candidates?.length > 0 && (
                <div className="card" style={{ marginBottom: 16, borderLeft: "3px solid var(--green)" }}>
                  <h3 style={{ fontSize: 14, color: "var(--green)", marginBottom: 8 }}>
                    HIGH CONFIDENCE PICKS ({scanResults.auto_bet_candidates.length})
                  </h3>
                  {scanResults.auto_bet_candidates.map((b, i) => (
                    <div key={i} className="scanner-pick strong">
                      <div className="pick-header">
                        <span className="pick-matchup">{b.p1_name} vs {b.p2_name}</span>
                        <span className={`badge badge-strong`}>STRONG</span>
                      </div>
                      <div className="pick-details">
                        <span>Predicted: <strong>{b.predicted_winner}</strong> ({b.confidence}%)</span>
                        <span>Edge: <strong>+{b.best_edge}%</strong></span>
                        <span>Odds: <strong>{b.best_odds}</strong></span>
                        <span>Kelly: {b.kelly_pct}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* All value bets */}
              <h3 style={{ fontSize: 14, color: "var(--text2)", marginBottom: 8 }}>All Value Bets</h3>
              {scanResults.value_bets.map((b, i) => (
                <div key={i} className={`scanner-pick ${b.rating.toLowerCase().replace(" ", "-")}`}>
                  <div className="pick-header">
                    <span className="pick-matchup">{b.p1_name} vs {b.p2_name}</span>
                    <span className={`badge badge-${b.rating === "STRONG" ? "strong" : b.rating === "GOOD" ? "good" : b.rating === "MILD" ? "mild" : "none"}`}>
                      {b.rating}
                    </span>
                  </div>
                  <div className="pick-details">
                    <span>Winner: <strong>{b.predicted_winner}</strong> ({b.confidence}%)</span>
                    <span>Bet: <strong>{b.bet_on_name}</strong></span>
                    <span>Edge: +{b.best_edge}%</span>
                    <span>Odds: {b.best_odds}</span>
                    <span>Kelly: {b.kelly_pct}%</span>
                    <span style={{ textTransform: "uppercase", fontSize: 11 }}>{b.tour}</span>
                  </div>
                </div>
              ))}

              {/* All predictions */}
              {scanResults.all_predictions?.length > 0 && (
                <>
                  <h3 style={{ fontSize: 14, color: "var(--text2)", margin: "16px 0 8px" }}>All Predictions</h3>
                  <table className="recent-table">
                    <thead>
                      <tr>
                        <th>Match</th>
                        <th>Predicted</th>
                        <th>Conf</th>
                        <th>Odds P1</th>
                        <th>Odds P2</th>
                        <th>Edge</th>
                        <th>Rating</th>
                      </tr>
                    </thead>
                    <tbody>
                      {scanResults.all_predictions.map((b, i) => (
                        <tr key={i}>
                          <td style={{ fontSize: 12 }}>{b.p1_name} vs {b.p2_name}</td>
                          <td><strong>{b.predicted_winner}</strong></td>
                          <td className={b.confidence >= 70 ? "form-hot" : b.confidence >= 60 ? "form-ok" : ""}>
                            {b.confidence}%
                          </td>
                          <td>{b.odds_p1}</td>
                          <td>{b.odds_p2}</td>
                          <td className={b.best_edge > 10 ? "form-hot" : b.best_edge > 5 ? "form-ok" : ""}>
                            {b.best_edge > 0 ? `+${b.best_edge}%` : `${b.best_edge}%`}
                          </td>
                          <td>
                            <span className={`badge badge-${b.rating === "STRONG" ? "strong" : b.rating === "GOOD" ? "good" : b.rating === "MILD" ? "mild" : "none"}`}>
                              {b.rating}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              )}
            </>
          )}
        </div>
      )}

      {/* Auto Bets Tab */}
      {tab === "auto-bets" && (
        <div>
          {autoBets.length === 0 ? (
            <div style={{ color: "var(--text2)", textAlign: "center", padding: 40 }}>
              No auto-bets yet. Run the scanner to auto-place bets on high-confidence picks.
            </div>
          ) : (
            <div className="bet-list">
              {autoBets.map((b) => (
                <div className="bet-item" key={b.id}>
                  <div>
                    <div className="matchup">
                      {b.p1_name} vs {b.p2_name}
                      <span className={`badge badge-${b.rating === "STRONG" ? "strong" : "good"}`} style={{ marginLeft: 8 }}>
                        {b.rating}
                      </span>
                    </div>
                    <div className="bet-details">
                      Bet: <strong>{b.bet_on_name}</strong> @ {b.odds} |
                      Stake: ${b.stake} |
                      Edge: {typeof b.edge === "number" ? (b.edge * 100).toFixed(1) : b.edge}% |
                      Conf: {typeof b.model_prob === "number" ? (b.model_prob * 100).toFixed(0) : b.model_prob}% |
                      Kelly: {b.kelly_pct}%
                      <span style={{ marginLeft: 6, fontSize: 11, color: "var(--accent)" }}>AUTO</span>
                    </div>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    {b.status === "pending" ? (
                      <div className="bet-actions">
                        <button className="btn-win" onClick={() => handleSettleAuto(b.id, true)}>Won</button>
                        <button className="btn-loss" onClick={() => handleSettleAuto(b.id, false)}>Lost</button>
                      </div>
                    ) : (
                      <div>
                        <span className={b.profit >= 0 ? "profit-pos" : "profit-neg"}>
                          {b.profit >= 0 ? "+" : ""}${b.profit?.toFixed(0)}
                        </span>
                        <div style={{ fontSize: 12, color: "var(--text2)" }}>
                          {b.won ? "WON" : "LOST"}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Performance Tab */}
      {tab === "performance" && performance && (
        <div>
          {performance.settled === 0 ? (
            <div style={{ color: "var(--text2)", textAlign: "center", padding: 40 }}>
              No settled bets yet. Settle some bets to see performance analysis.
            </div>
          ) : (
            <>
              {/* Calibration */}
              {performance.calibration && Object.keys(performance.calibration).length > 0 && (
                <div className="card" style={{ marginBottom: 16 }}>
                  <h3 style={{ fontSize: 14, color: "var(--text2)", marginBottom: 8 }}>Model Calibration</h3>
                  <p style={{ fontSize: 12, color: "var(--text2)", marginBottom: 8 }}>
                    Is the model's confidence accurate? Calibration error shows predicted vs actual win rate.
                  </p>
                  <table className="recent-table">
                    <thead>
                      <tr>
                        <th>Confidence</th>
                        <th>Bets</th>
                        <th>Actual Win%</th>
                        <th>Avg Conf</th>
                        <th>Cal. Error</th>
                        <th>ROI</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(performance.calibration).map(([bucket, data]) => (
                        <tr key={bucket}>
                          <td>{bucket}</td>
                          <td>{data.count}</td>
                          <td className={data.actual_win_rate >= 55 ? "form-hot" : "form-cold"}>
                            {data.actual_win_rate}%
                          </td>
                          <td>{data.avg_confidence}%</td>
                          <td className={Math.abs(data.calibration_error) < 5 ? "form-hot" : "form-cold"}>
                            {data.calibration_error > 0 ? "+" : ""}{data.calibration_error}%
                          </td>
                          <td className={data.roi >= 0 ? "profit-pos" : "profit-neg"}>
                            {data.roi > 0 ? "+" : ""}{data.roi}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Edge Analysis */}
              {performance.edge_analysis && Object.keys(performance.edge_analysis).length > 0 && (
                <div className="card" style={{ marginBottom: 16 }}>
                  <h3 style={{ fontSize: 14, color: "var(--text2)", marginBottom: 8 }}>Edge Analysis</h3>
                  <table className="recent-table">
                    <thead>
                      <tr>
                        <th>Edge Bucket</th>
                        <th>Bets</th>
                        <th>Win Rate</th>
                        <th>Profit</th>
                        <th>ROI</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(performance.edge_analysis).map(([bucket, data]) => (
                        <tr key={bucket}>
                          <td>{bucket}</td>
                          <td>{data.count}</td>
                          <td className={data.win_rate >= 55 ? "form-hot" : "form-cold"}>
                            {data.win_rate}%
                          </td>
                          <td className={data.profit >= 0 ? "profit-pos" : "profit-neg"}>
                            ${data.profit}
                          </td>
                          <td className={data.roi >= 0 ? "profit-pos" : "profit-neg"}>
                            {data.roi > 0 ? "+" : ""}{data.roi}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Auto vs Manual */}
              {performance.auto_performance && performance.auto_performance.count > 0 && (
                <div className="card" style={{ marginBottom: 16 }}>
                  <h3 style={{ fontSize: 14, color: "var(--text2)", marginBottom: 8 }}>Auto vs Manual Bets</h3>
                  <div className="bets-summary">
                    <div className="stat-box">
                      <div className="label">Auto Win Rate</div>
                      <div className="val" style={{ color: performance.auto_performance.win_rate >= 55 ? "var(--green)" : "var(--red)" }}>
                        {performance.auto_performance.win_rate}%
                      </div>
                    </div>
                    <div className="stat-box">
                      <div className="label">Auto ROI</div>
                      <div className="val" style={{ color: performance.auto_performance.roi >= 0 ? "var(--green)" : "var(--red)" }}>
                        {performance.auto_performance.roi > 0 ? "+" : ""}{performance.auto_performance.roi}%
                      </div>
                    </div>
                    <div className="stat-box">
                      <div className="label">Auto Profit</div>
                      <div className="val" style={{ color: performance.auto_performance.profit >= 0 ? "var(--green)" : "var(--red)" }}>
                        ${performance.auto_performance.profit}
                      </div>
                    </div>
                    <div className="stat-box">
                      <div className="label">Auto Bets</div>
                      <div className="val">{performance.auto_performance.count}</div>
                    </div>
                  </div>
                </div>
              )}

              {/* Recommendations */}
              {performance.recommendations?.length > 0 && (
                <div className="card" style={{ borderLeft: "3px solid var(--yellow)" }}>
                  <h3 style={{ fontSize: 14, color: "var(--yellow)", marginBottom: 8 }}>Recommendations</h3>
                  {performance.recommendations.map((r, i) => (
                    <div key={i} style={{ fontSize: 13, color: "var(--text2)", marginBottom: 4 }}>
                      {r}
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ── Main App ──
export default function App() {
  const [page, setPage] = useState("dashboard");
  const [navPlayerId, setNavPlayerId] = useState(null);

  const handlePlayerClick = (playerId, playerName) => {
    setNavPlayerId({ player_id: playerId, name: playerName });
    setPage("player");
  };

  return (
    <>
      <header className="app-header">
        <h1>Tennis <span>Predictor</span></h1>
        <nav>
          <button className={page === "dashboard" ? "active" : ""} onClick={() => setPage("dashboard")}>
            Dashboard
          </button>
          <button className={page === "predict" ? "active" : ""} onClick={() => setPage("predict")}>
            Predict
          </button>
          <button className={page === "live" ? "active" : ""} onClick={() => setPage("live")}>
            Live
          </button>
          <button className={page === "bets" ? "active" : ""} onClick={() => setPage("bets")}>
            Paper Bets
          </button>
          <button className={page === "player" ? "active" : ""} onClick={() => setPage("player")}>
            Players
          </button>
        </nav>
      </header>

      {page === "dashboard" && <DashboardPage />}
      {page === "predict" && <PredictPage />}
      {page === "live" && <LivePage onPlayerClick={handlePlayerClick} />}
      {page === "bets" && <PaperBetsPage />}
      {page === "player" && <PlayerPage initialPlayer={navPlayerId} onClear={() => setNavPlayerId(null)} />}
    </>
  );
}
