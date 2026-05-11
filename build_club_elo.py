"""
Phase 12: Build club ELO + squad club-form feature.

Steps:
  1. Fetch current ClubElo data (~630 clubs across Europe)
  2. Save as `club_elo` table
  3. Match each FIFA player's club_name to a ClubElo team (fuzzy match)
  4. For each WC2026 team, compute squad_club_elo = mean of top-25 players' club ELOs
  5. Save as `squad_club_form` table
"""
import sys, io, sqlite3, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import soccerdata as sd
from fuzzywuzzy import process, fuzz

DB_PATH = Path(__file__).parent / "data" / "soccer.db"


# -------------------------------------------------------------------------
# 1. Fetch ClubElo
# -------------------------------------------------------------------------
def fetch_club_elo(date: str = "2026-05-10") -> pd.DataFrame:
    print(f">> Fetching ClubElo as of {date}")
    elo = sd.ClubElo()
    df = elo.read_by_date(date).reset_index()
    df = df.rename(columns={"team": "club_name"})
    print(f"  got {len(df)} clubs (top ELO: {df['elo'].max():.0f}, median: {df['elo'].median():.0f})")
    return df[["club_name", "country", "level", "elo", "league"]]


# -------------------------------------------------------------------------
# 2. Fuzzy match FIFA club_name → ClubElo name
# -------------------------------------------------------------------------
def build_club_match_map(fifa_clubs: list[str], club_elo_names: list[str],
                          threshold: int = 75) -> dict[str, str | None]:
    """Returns {fifa_club: clubelo_club or None}."""
    out: dict[str, str | None] = {}
    for fc in set(fifa_clubs):
        if not isinstance(fc, str) or not fc.strip():
            out[fc] = None
            continue
        best = process.extractOne(fc, club_elo_names, scorer=fuzz.WRatio)
        out[fc] = best[0] if best and best[1] >= threshold else None
    return out


# -------------------------------------------------------------------------
# 3. Squad club form per WC team
# -------------------------------------------------------------------------
def compute_squad_club_form(conn, club_elo_df: pd.DataFrame) -> pd.DataFrame:
    # Get player → club mapping for top players of each WC team
    aliases = pd.read_sql("SELECT canonical, fifa_name FROM nation_aliases", conn)
    alias_map = dict(zip(aliases["canonical"], aliases["fifa_name"]))

    teams = pd.read_sql("SELECT team FROM wc2026_teams", conn)["team"].tolist()
    fifa_clubs = set()
    for t in teams:
        fifa_name = alias_map.get(t, t)
        sub = pd.read_sql(
            "SELECT club_name FROM players "
            "WHERE LOWER(nationality_name) = LOWER(?) "
            "ORDER BY overall DESC LIMIT 25",
            conn, params=(fifa_name,),
        )
        fifa_clubs.update(sub["club_name"].dropna().tolist())

    print(f">> Fuzzy matching {len(fifa_clubs)} unique FIFA clubs to ClubElo ({len(club_elo_df)} clubs)...")
    club_match = build_club_match_map(list(fifa_clubs), club_elo_df["club_name"].tolist(), threshold=75)
    matched_count = sum(1 for v in club_match.values() if v is not None)
    print(f"  matched: {matched_count}/{len(fifa_clubs)} ({matched_count/max(len(fifa_clubs),1)*100:.1f}%)")

    elo_lookup = dict(zip(club_elo_df["club_name"], club_elo_df["elo"]))

    rows = []
    for t in teams:
        fifa_name = alias_map.get(t, t)
        sub = pd.read_sql(
            "SELECT short_name, club_name, overall FROM players "
            "WHERE LOWER(nationality_name) = LOWER(?) "
            "ORDER BY overall DESC LIMIT 25",
            conn, params=(fifa_name,),
        )
        if len(sub) == 0:
            rows.append({"team": t, "n_players_matched": 0,
                         "squad_club_elo": np.nan, "club_coverage": 0.0})
            continue
        club_elos = []
        for cn in sub["club_name"]:
            mapped = club_match.get(cn)
            if mapped and mapped in elo_lookup:
                club_elos.append(elo_lookup[mapped])
        n_matched = len(club_elos)
        rows.append({
            "team": t,
            "n_players_total": len(sub),
            "n_players_matched": n_matched,
            "club_coverage": round(n_matched / len(sub), 2),
            "squad_club_elo": round(np.mean(club_elos), 1) if club_elos else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        # Step 1+2: fetch and save club ELO
        ce = fetch_club_elo("2026-05-10")
        ce.to_sql("club_elo", conn, if_exists="replace", index=False)
        print(f"  saved: club_elo ({len(ce)} rows)\n")

        # Step 3: compute squad club form
        scf = compute_squad_club_form(conn, ce)
        scf.to_sql("squad_club_form", conn, if_exists="replace", index=False)
        print(f"\nsaved: squad_club_form ({len(scf)} rows)\n")

        # Show ranking
        ranked = scf.sort_values("squad_club_elo", ascending=False, na_position="last")
        print("WC2026 teams ranked by squad club ELO:")
        print(ranked.to_string(index=False))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
