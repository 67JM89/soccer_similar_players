"""
Phase B — Scrape Understat player-season xG for 10 seasons × 5 European leagues,
match to FIFA player table, compute national-team xG aggregates per (team, year).

Output tables:
  - xg_player_seasons: (player, season_year, league, games, npxG, xA, xG, npg, xG90)
  - team_xg_year: (team, year, sum_xG, sum_xA, sum_npxG, top11_xG, n_players)
  - point-in-time match feature joined later

Run once; soccerdata caches results locally.
"""
import sys, io, sqlite3, time
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
SEASONS = list(range(2014, 2025))  # 2014-15 ... 2024-25


def fetch_player_seasons() -> pd.DataFrame:
    """Combine player-season stats across leagues & seasons."""
    frames = []
    for season in SEASONS:
        print(f"  fetching season {season}...")
        try:
            us = sd.Understat(leagues=LEAGUES, seasons=[season])
            df = us.read_player_season_stats()
            df = df.reset_index()
            df["season_year"] = season
            frames.append(df)
            print(f"    → {len(df):,} rows")
        except Exception as e:
            print(f"    ⚠ skipped: {e}")
        time.sleep(0.3)
    out = pd.concat(frames, ignore_index=True)
    # Keep relevant columns (lowercase per soccerdata convention)
    keep = [c for c in ["player","player_id","team","matches","minutes","goals","xg","xa","np_goals","np_xg","season_year","league"] if c in out.columns]
    return out[keep]


def match_xg_to_fifa(xg: pd.DataFrame) -> pd.DataFrame:
    """For each unique xG player, find best fuzzy match to a FIFA player.
    Returns DataFrame with column `fifa_short_name`."""
    conn = sqlite3.connect(DB)
    fifa = pd.read_sql("""
        SELECT DISTINCT short_name, long_name, nationality_name
        FROM players WHERE short_name IS NOT NULL
    """, conn)
    conn.close()

    # Build choice list with index → (short, nationality)
    choices = (fifa["long_name"].fillna("") + " | " + fifa["short_name"]).tolist()
    idx_to_short = dict(enumerate(fifa["short_name"].tolist()))

    unique_xg = xg.groupby("player").size().reset_index(name="rows")
    unique_xg = unique_xg.sort_values("rows", ascending=False)
    print(f"\n  Matching {len(unique_xg):,} unique Understat players → FIFA…")

    results = []
    for i, name in enumerate(unique_xg["player"].tolist()):
        if i % 500 == 0 and i > 0:
            print(f"    {i}/{len(unique_xg)}…")
        best = process.extractOne(name, choices, scorer=fuzz.WRatio, score_cutoff=85)
        if best:
            results.append({"player": name, "fifa_short_name": idx_to_short[best[2]], "score": int(best[1])})
        else:
            results.append({"player": name, "fifa_short_name": None, "score": 0})
    match_df = pd.DataFrame(results)
    print(f"  matched: {match_df['fifa_short_name'].notna().sum():,} / {len(match_df):,}")
    return match_df


def aggregate_team_year(xg: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """For each (team, season_year), aggregate top-11 xG of national-team players."""
    conn = sqlite3.connect(DB)
    al = pd.read_sql("SELECT canonical, fifa_name FROM nation_aliases", conn)
    alias = dict(zip(al["canonical"], al["fifa_name"]))
    fifa_to_nat = pd.read_sql("""
        SELECT DISTINCT short_name, nationality_name FROM players
        WHERE short_name IS NOT NULL
    """, conn).drop_duplicates("short_name")
    conn.close()

    # Join xG with match → fifa_short → nationality
    xg = xg.merge(matches, on="player", how="left")
    xg = xg.merge(
        fifa_to_nat.rename(columns={"short_name":"fifa_short_name", "nationality_name":"player_nation"}),
        on="fifa_short_name", how="left",
    )
    xg = xg.dropna(subset=["player_nation"])

    # Compute aggregates per (player_nation, year), top 11 by xG
    print(f"\n  Aggregating team-year xG (top 11)…")
    rows = []
    for (nat, year), grp in xg.groupby(["player_nation", "season_year"]):
        top11 = grp.sort_values("xg", ascending=False).head(11)
        rows.append({
            "nation": nat,
            "year": int(year),
            "n_xg_players": len(grp),
            "n_top11": len(top11),
            "sum_xG_top11": float(top11["xg"].sum()),
            "sum_xA_top11": float(top11["xa"].sum()) if "xa" in top11.columns else 0.0,
            "sum_npxG_top11": float(top11["np_xg"].sum()) if "np_xg" in top11.columns else 0.0,
            "mean_xG_top11": float(top11["xg"].mean()),
            "total_minutes_top11": float(top11["minutes"].sum()) if "minutes" in top11.columns else 0.0,
        })
    out = pd.DataFrame(rows)
    print(f"  {len(out):,} (nation, year) rows")
    return out


def main():
    print("=== Phase B: Understat xG scraping (2014-2024, 5 leagues) ===\n")
    print("Step 1/3: Fetch player-season stats")
    xg = fetch_player_seasons()
    print(f"\nTotal xG rows: {len(xg):,}")

    print("\nStep 2/3: Match to FIFA players")
    matches = match_xg_to_fifa(xg)

    print("\nStep 3/3: Aggregate per (nation, year)")
    agg = aggregate_team_year(xg, matches)

    # Write all tables
    conn = sqlite3.connect(DB)
    xg.to_sql("xg_player_seasons", conn, if_exists="replace", index=False)
    matches.to_sql("xg_player_link", conn, if_exists="replace", index=False)
    agg.to_sql("team_xg_year", conn, if_exists="replace", index=False)
    conn.commit()

    # Summary
    print("\n=== Summary ===")
    print(f"  xg_player_seasons : {len(xg):,} rows")
    print(f"  xg_player_link    : {len(matches):,} rows ({matches['fifa_short_name'].notna().sum():,} matched)")
    print(f"  team_xg_year      : {len(agg):,} rows")
    print()
    print("Latest year (2024) top-10 by sum_xG_top11:")
    print(agg[agg["year"] == 2024].nlargest(10, "sum_xG_top11")[
        ["nation","year","n_xg_players","sum_xG_top11","mean_xG_top11"]
    ].to_string(index=False))
    conn.close()


if __name__ == "__main__":
    main()
