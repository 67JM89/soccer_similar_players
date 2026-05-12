"""
Match Transfermarkt players (tm_squads) to FIFA players (players table).
Writes table `tm_player_link`: (tm_player_id, fifa_short_name, team, score).

Strategy:
  - Filter FIFA players by team nationality (via nation_aliases)
  - Within nationality, fuzzy match on long_name OR short_name
  - Accept only matches with score >= 85
"""
import sys, io, sqlite3
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
import pandas as pd
from rapidfuzz import fuzz, process

DB = Path(__file__).parent / "data" / "soccer.db"


def main():
    conn = sqlite3.connect(DB)
    tm = pd.read_sql("SELECT * FROM tm_squads", conn)
    al = pd.read_sql("SELECT canonical, fifa_name FROM nation_aliases", conn)
    alias = dict(zip(al["canonical"], al["fifa_name"]))

    print(f"Loaded {len(tm):,} TM player rows across {tm['team'].nunique()} teams")
    matches = []
    for team in sorted(tm["team"].unique()):
        fifa_name = alias.get(team, team)
        fifa = pd.read_sql("""
            SELECT short_name, long_name, club_name, overall
            FROM players WHERE LOWER(nationality_name) = LOWER(?)
            ORDER BY overall DESC LIMIT 60
        """, conn, params=(fifa_name,))
        if len(fifa) == 0:
            continue
        fifa_names = (fifa["long_name"].fillna("") + " | " + fifa["short_name"].fillna("")).tolist()
        idx_to_short = dict(enumerate(fifa["short_name"].tolist()))

        sub = tm[tm["team"] == team]
        n_matched = 0
        for _, row in sub.iterrows():
            tm_name = row["name"]
            best = process.extractOne(
                tm_name, fifa_names, scorer=fuzz.WRatio, score_cutoff=80
            )
            if best:
                matched_text, score, idx = best
                fifa_short = idx_to_short[idx]
                matches.append({
                    "tm_player_id": int(row["tm_player_id"]),
                    "tm_name": tm_name,
                    "team": team,
                    "fifa_short_name": fifa_short,
                    "score": int(score),
                })
                n_matched += 1
        print(f"  {team:25s}: {n_matched}/{len(sub)} matched")

    df = pd.DataFrame(matches)
    print(f"\nTotal matched: {len(df):,} / {len(tm):,}  ({100*len(df)/max(len(tm),1):.1f}%)")

    conn.execute("DROP TABLE IF EXISTS tm_player_link")
    conn.execute("""
        CREATE TABLE tm_player_link (
            tm_player_id INTEGER PRIMARY KEY,
            tm_name TEXT, team TEXT, fifa_short_name TEXT, score INTEGER
        )
    """)
    df.to_sql("tm_player_link", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    print(f"\nWrote tm_player_link ({len(df):,} rows)")


if __name__ == "__main__":
    main()
