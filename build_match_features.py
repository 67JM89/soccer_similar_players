"""
Phase 4a: Point-in-time match features for ALL 49K historical matches.

Replays matches chronologically; before applying each result, captures
the *current* ELO of both teams. This avoids the look-ahead leak that
would happen if we used today's ELO as a feature for past matches.

Output: table `match_features` with columns:
  date, home_team, away_team, neutral, tournament,
  elo_home_pre, elo_away_pre, elo_diff, home_advantage,
  home_score, away_score, result  (1=home win, 0=draw, -1=away win)
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

DB_PATH = Path(__file__).parent / "data" / "soccer.db"


# Same K logic as build_database.py — keep ELO consistent
def k_weight(tourn: str) -> int:
    t = (tourn or "").lower()
    if "world cup" in t and "qualif" not in t: return 60
    if "qualif" in t: return 40
    if any(x in t for x in ["uefa euro", "copa amér", "copa america",
                             "africa cup", "afc asian", "concacaf"]):
        return 50
    if "friendly" in t: return 20
    return 30


def build():
    conn = sqlite3.connect(DB_PATH)
    matches = pd.read_sql(
        "SELECT date, home_team, away_team, home_score, away_score, "
        "tournament, neutral FROM intl_matches ORDER BY date",
        conn,
    )
    matches["date"] = pd.to_datetime(matches["date"])
    print(f"matches loaded: {len(matches):,}")

    INIT, BONUS = 1500.0, 100.0
    elo: dict[str, float] = {}

    rows = []
    for r in matches.itertuples(index=False):
        h, a = r.home_team, r.away_team
        eh = elo.get(h, INIT)
        ea = elo.get(a, INIT)

        # snapshot pre-match
        ha = BONUS if r.neutral == 0 else 0
        rows.append({
            "date": r.date,
            "home_team": h, "away_team": a,
            "neutral": r.neutral,
            "tournament": r.tournament,
            "elo_home_pre": eh,
            "elo_away_pre": ea,
            "elo_diff": (eh + ha) - ea,   # includes home bonus
            "home_advantage": ha,
            "home_score": r.home_score,
            "away_score": r.away_score,
        })

        # update ELO with result
        eh_adj = eh + ha
        exp_h = 1.0 / (1.0 + 10 ** ((ea - eh_adj) / 400))
        if r.home_score > r.away_score:   act_h = 1.0
        elif r.home_score < r.away_score: act_h = 0.0
        else:                             act_h = 0.5
        gd = abs(r.home_score - r.away_score)
        if gd <= 1:    g = 1.0
        elif gd == 2:  g = 1.5
        else:          g = (11.0 + gd) / 8.0
        K = k_weight(r.tournament) * g
        delta = K * (act_h - exp_h)
        elo[h] = eh + delta
        elo[a] = ea - delta

    df = pd.DataFrame(rows)
    df["result"] = np.sign(df["home_score"] - df["away_score"]).astype(int)
    df["goal_diff"] = df["home_score"] - df["away_score"]
    df["total_goals"] = df["home_score"] + df["away_score"]

    df.to_sql("match_features", conn, if_exists="replace", index=False)
    conn.close()

    print(f"saved: match_features ({len(df):,} rows)")
    print()
    print("Recent 5 rows:")
    print(df.tail(5)[["date","home_team","away_team","elo_home_pre",
                       "elo_away_pre","home_score","away_score"]].to_string(index=False))
    print()
    print("Result distribution:")
    print(df["result"].value_counts(normalize=True).round(3).rename({1:"home_win",0:"draw",-1:"away_win"}))


if __name__ == "__main__":
    build()
