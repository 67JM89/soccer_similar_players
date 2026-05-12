"""
Probe coverage: scrape Understat player season stats for top 5 leagues,
then check how many WC2026 qualified-team players are covered.

Decision criterion:
  - If >=70% of top-23 players per WC team have xG data → proceed to full pipeline
  - Else → pivot to a different data source
"""
import sys, io, sqlite3
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
import pandas as pd
import soccerdata as sd
from rapidfuzz import fuzz, process

DB = Path(__file__).parent / "data" / "soccer.db"
LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]
SEASON = 2024  # 2024-25 most recent complete


def fetch_xg() -> pd.DataFrame:
    print(f"\n=== Fetching Understat player stats ({len(LEAGUES)} leagues, season {SEASON}) ===")
    us = sd.Understat(leagues=LEAGUES, seasons=[SEASON])
    df = us.read_player_season_stats()
    print(f"  rows: {len(df):,}  | unique players: {df['player_id'].nunique():,}")
    return df.reset_index()


def load_wc_roster() -> pd.DataFrame:
    conn = sqlite3.connect(DB)
    al = pd.read_sql("SELECT canonical, fifa_name FROM nation_aliases", conn)
    alias = dict(zip(al["canonical"], al["fifa_name"]))
    teams = pd.read_sql('SELECT DISTINCT team FROM wc2026_groups', conn)["team"].tolist()
    rows = []
    for team in teams:
        fifa = alias.get(team, team)
        df = pd.read_sql("""
            SELECT short_name, long_name, club_name, overall, is_gk
            FROM players WHERE LOWER(nationality_name) = LOWER(?)
            ORDER BY overall DESC LIMIT 23
        """, conn, params=(fifa,))
        df["team"] = team
        rows.append(df)
    conn.close()
    out = pd.concat(rows, ignore_index=True)
    print(f"\n=== WC2026 rosters loaded ===  {len(out):,} player-rows across {out['team'].nunique()} teams")
    return out


def match_players(xg: pd.DataFrame, fifa: pd.DataFrame) -> pd.DataFrame:
    """Fuzzy-match Understat player names to FIFA player names.
    For each FIFA player (per team), find best Understat match.
    """
    xg_names = xg["player"].dropna().unique().tolist()
    print(f"\n=== Matching {len(fifa):,} FIFA players against {len(xg_names):,} Understat names ===")
    results = []
    for _, row in fifa.iterrows():
        target_names = [n for n in [row["short_name"], row["long_name"]] if pd.notna(n)]
        best_score = 0
        best_match = None
        for t in target_names:
            m = process.extractOne(t, xg_names, scorer=fuzz.WRatio, score_cutoff=85)
            if m and m[1] > best_score:
                best_score, best_match = m[1], m[0]
        results.append({
            "team": row["team"],
            "fifa_name": row["short_name"],
            "overall": row["overall"],
            "is_gk": row["is_gk"],
            "matched_understat": best_match,
            "score": best_score,
        })
    out = pd.DataFrame(results)
    matched = out[out["matched_understat"].notna()]
    print(f"  matched: {len(matched):,} / {len(out):,}  ({100*len(matched)/len(out):.1f}%)")
    return out


def coverage_report(matches: pd.DataFrame) -> None:
    print("\n=== Per-team coverage (top-23 squad) ===")
    by_team = matches.groupby("team").agg(
        total=("fifa_name", "count"),
        matched=("matched_understat", lambda s: s.notna().sum()),
    )
    by_team["pct"] = (100 * by_team["matched"] / by_team["total"]).round(1)
    by_team = by_team.sort_values("pct", ascending=False)
    print(by_team.to_string())
    n_good = (by_team["pct"] >= 70).sum()
    n_med = ((by_team["pct"] >= 50) & (by_team["pct"] < 70)).sum()
    n_bad = (by_team["pct"] < 50).sum()
    print(f"\n=== Summary ===")
    print(f"  Teams ≥70% covered: {n_good} / {len(by_team)}")
    print(f"  Teams 50-69%:        {n_med} / {len(by_team)}")
    print(f"  Teams <50%:          {n_bad} / {len(by_team)}")
    overall = (matches["matched_understat"].notna().sum() / len(matches)) * 100
    print(f"  Overall match rate:  {overall:.1f}%")


if __name__ == "__main__":
    xg = fetch_xg()
    fifa = load_wc_roster()
    matches = match_players(xg, fifa)
    coverage_report(matches)
