"""
Phase 7a: Recent-form features per match (point-in-time).

For every historical match, computes pre-match form of BOTH teams over
their previous N (default 10) international matches:
  - form_ppg : points per game (W=3, D=1, L=0)
  - form_gd  : average goal differential
  - form_cs  : clean sheet rate (% with GA == 0)

Output:
  - table `match_form`     : per-match snapshot, joined to match_features by (date, home_team, away_team)
  - table `current_form`   : latest form for every team (used at predict time for 2026)
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import sqlite3
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "soccer.db"
WINDOW = 10
MIN_MATCHES = 5  # need at least 5 prior matches for valid form


def main():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT date, home_team, away_team, home_score, away_score "
        "FROM intl_matches ORDER BY date",
        conn,
    )
    df["date"] = pd.to_datetime(df["date"])
    print(f"matches: {len(df):,}")

    hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=WINDOW))

    def snapshot(team: str):
        h = hist[team]
        if len(h) < MIN_MATCHES:
            return np.nan, np.nan, np.nan
        return (
            float(np.mean([m[0] for m in h])),         # ppg
            float(np.mean([m[1] for m in h])),         # gd
            float(np.mean([1.0 if m[2] == 0 else 0.0 for m in h])),  # clean sheet
        )

    rows = []
    for r in df.itertuples(index=False):
        h, a = r.home_team, r.away_team
        h_ppg, h_gd, h_cs = snapshot(h)
        a_ppg, a_gd, a_cs = snapshot(a)
        rows.append({
            "date": r.date, "home_team": h, "away_team": a,
            "h_form_ppg": h_ppg, "h_form_gd": h_gd, "h_form_cs": h_cs,
            "a_form_ppg": a_ppg, "a_form_gd": a_gd, "a_form_cs": a_cs,
        })

        # update history with THIS match (post-match)
        hs, as_ = r.home_score, r.away_score
        if hs > as_:   h_pts, a_pts = 3, 0
        elif hs < as_: h_pts, a_pts = 0, 3
        else:          h_pts, a_pts = 1, 1
        hist[h].append((h_pts, hs - as_, as_))   # (points, GD, GA)
        hist[a].append((a_pts, as_ - hs, hs))

    form_df = pd.DataFrame(rows)
    form_df.to_sql("match_form", conn, if_exists="replace", index=False)
    print(f"saved: match_form ({len(form_df):,} rows)")

    # Current form: snapshot for every team after all matches
    cur_rows = []
    for team in hist:
        ppg, gd, cs = snapshot(team)
        cur_rows.append({
            "team": team,
            "form_ppg": ppg, "form_gd": gd, "form_cs": cs,
            "matches_seen": len(hist[team]),
        })
    cur_df = pd.DataFrame(cur_rows).sort_values("form_ppg", ascending=False, na_position="last")
    cur_df.to_sql("current_form", conn, if_exists="replace", index=False)
    print(f"saved: current_form ({len(cur_df)} teams)")

    # Sanity: top-10 current form
    print("\nTop 10 by current form (PPG):")
    print(cur_df.head(10).to_string(index=False))
    print("\nForm for major WC2026 teams:")
    big = ["Spain","Argentina","France","England","Brazil","Germany","Mexico","United States","Canada","Japan","Morocco"]
    print(cur_df[cur_df["team"].isin(big)].to_string(index=False))

    conn.close()


if __name__ == "__main__":
    main()
