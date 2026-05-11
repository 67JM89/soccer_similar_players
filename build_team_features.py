"""
Phase 3: Team-level features for WC2026 teams.

Creates table `team_features` with columns:
  - team                : canonical name (WC2026 list)
  - elo                 : current ELO (from team_elo)
  - elo_rank
  - squad_overall       : mean FIFA overall of top-25 players
  - squad_atk/def/phy   : mean of derived 12-stat scores
  - squad_age           : mean age of top-25
  - form_points_per_game: PPG over last 20 matches
  - form_gd_per_game    : avg goal differential over last 20 matches
  - form_clean_sheet_pct: % of last-20 with no goals conceded
  - form_matches        : actual count (may be < 20 for small nations)
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
DB_PATH = ROOT / "data" / "soccer.db"


def alias_map(conn):
    """Return dict {canonical_wc_name: fifa_name} for known aliases.
    Default behavior: use canonical name itself if no alias is set."""
    df = pd.read_sql("SELECT * FROM nation_aliases", conn)
    return dict(zip(df["canonical"], df["fifa_name"]))


# -------------------------------------------------------------------
# Squad rating: top-25 by overall per WC2026 team
# -------------------------------------------------------------------
def squad_features(conn, top_n: int = 25) -> pd.DataFrame:
    aliases = alias_map(conn)
    teams = pd.read_sql("SELECT team FROM wc2026_teams", conn)

    rows = []
    for t in teams["team"]:
        fifa_name = aliases.get(t, t)
        sub = pd.read_sql(
            "SELECT overall, age, score_atk, score_def, score_phy, score_gk_save "
            "FROM players WHERE LOWER(nationality_name) = LOWER(?) "
            "ORDER BY overall DESC LIMIT ?",
            conn, params=(fifa_name, top_n),
        )
        if len(sub) == 0:
            rows.append({
                "team": t, "squad_size_in_fifa": 0,
                "squad_overall": np.nan, "squad_atk": np.nan,
                "squad_def": np.nan, "squad_phy": np.nan,
                "squad_age": np.nan, "squad_top11_overall": np.nan,
            })
            continue
        # Use top-N for squad mean, but separately compute top-11 overall (likely starters)
        top11 = sub.head(11)["overall"].mean()
        rows.append({
            "team": t,
            "squad_size_in_fifa": len(sub),
            "squad_overall": sub["overall"].mean().round(2),
            "squad_atk":     sub["score_atk"].mean().round(2),
            "squad_def":     sub["score_def"].mean().round(2),
            "squad_phy":     sub["score_phy"].mean().round(2),
            "squad_age":     sub["age"].mean().round(2),
            "squad_top11_overall": round(top11, 2),
        })
    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Recent form: last 20 intl matches per team
# -------------------------------------------------------------------
def form_features(conn, last_n: int = 20) -> pd.DataFrame:
    matches = pd.read_sql("SELECT * FROM intl_matches ORDER BY date", conn)
    matches["date"] = pd.to_datetime(matches["date"])
    aliases = alias_map(conn)
    # use 'intl_name' alias if explicitly given, else canonical
    name_in_intl = {row["canonical"]: (row["intl_name"] or row["canonical"])
                    for _, row in pd.read_sql("SELECT canonical, intl_name FROM nation_aliases", conn).iterrows()}

    teams = pd.read_sql("SELECT team FROM wc2026_teams", conn)["team"].tolist()
    rows = []
    for t in teams:
        intl_t = name_in_intl.get(t, t)
        m = matches[(matches["home_team"] == intl_t) | (matches["away_team"] == intl_t)].tail(last_n)
        if len(m) == 0:
            rows.append({"team": t, "form_matches": 0, "form_points_per_game": np.nan,
                         "form_gd_per_game": np.nan, "form_clean_sheet_pct": np.nan})
            continue
        is_home = m["home_team"] == intl_t
        gf = np.where(is_home, m["home_score"], m["away_score"])
        ga = np.where(is_home, m["away_score"], m["home_score"])
        result = np.where(gf > ga, 3, np.where(gf == ga, 1, 0))
        rows.append({
            "team": t,
            "form_matches": len(m),
            "form_points_per_game": round(result.mean(), 3),
            "form_gd_per_game":     round((gf - ga).mean(), 3),
            "form_clean_sheet_pct": round((ga == 0).mean(), 3),
        })
    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Combine + save
# -------------------------------------------------------------------
def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        print(">> squad_features (top-25 FIFA players per WC team)")
        sq = squad_features(conn, top_n=25)

        print(">> form_features (last 20 international matches per WC team)")
        fm = form_features(conn, last_n=20)

        elo = pd.read_sql("SELECT team, elo, rank AS elo_rank FROM team_elo", conn)

        feats = (sq.merge(fm, on="team", how="left")
                   .merge(elo, on="team", how="left"))
        feats = feats[[
            "team", "elo", "elo_rank",
            "squad_size_in_fifa", "squad_overall", "squad_top11_overall",
            "squad_atk", "squad_def", "squad_phy", "squad_age",
            "form_matches", "form_points_per_game",
            "form_gd_per_game", "form_clean_sheet_pct",
        ]]
        feats.to_sql("team_features", conn, if_exists="replace", index=False)

        print(f"\nsaved: team_features ({len(feats)} teams, {len(feats.columns)} cols)\n")

        # display by composite ranking
        view = feats.sort_values("elo", ascending=False).reset_index(drop=True)
        print("Top 15 by ELO:")
        print(view.head(15).to_string(index=False))
        print("\nBottom 5 by ELO:")
        print(view.tail(5).to_string(index=False))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
