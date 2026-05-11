"""
Phase 2: Raw data -> unified SQLite (data/soccer.db)

Tables created:
  - players          : FIFA 23 latest ratings + derived 12-stat (atk4/def4/phy4)
  - intl_matches     : 49K international matches (1872-now)
  - team_elo         : current ELO per national team (computed from intl_matches)
  - wc2026_teams     : 48 qualified teams
  - wc2026_schedule  : matchday calendar
  - wc2026_venues    : 16 venues
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "soccer.db"


def step(msg):
    print(f"\n{'='*60}\n>> {msg}\n{'='*60}")


# ========================================================================
# 1) PLAYERS — FIFA 23 (most recent FIFA edition in our data)
# ========================================================================
def build_players(conn):
    step("1/6  players  (FIFA 23 ratings + 12 derived stats)")

    df = pd.read_csv(RAW / "fifa23" / "male_players (legacy).csv", low_memory=False)
    print(f"  raw rows: {len(df):,}")

    # Keep only FIFA 23 (latest version) rows
    df = df[df["fifa_version"] == 23].copy()
    # Among multiple updates per player, keep the most recent update
    df["fifa_update_date"] = pd.to_datetime(df["fifa_update_date"], errors="coerce")
    df = df.sort_values("fifa_update_date").drop_duplicates("player_id", keep="last")
    print(f"  fifa23 latest update: {len(df):,} unique players")

    # 12 derived stats — user spec: 4 atk + 4 def + 4 phy
    df["atk_finishing"]      = df["attacking_finishing"]
    df["atk_shooting"]       = df["power_shot_power"]
    df["atk_dribbling"]      = df["dribbling"]
    df["atk_vision"]         = df["mentality_vision"]

    df["def_tackling"]       = df["defending_standing_tackle"]
    df["def_marking"]        = df["defending_marking_awareness"]
    df["def_interceptions"]  = df["mentality_interceptions"]
    df["def_aggression"]     = df["mentality_aggression"]

    df["phy_pace"]           = df["pace"]
    df["phy_strength"]       = df["power_strength"]
    df["phy_stamina"]        = df["power_stamina"]
    df["phy_jumping"]        = df["power_jumping"]

    # GK flag + GK 4-stat replacements (Distribution / Saves)
    df["is_gk"] = df["player_positions"].fillna("").str.contains("GK").astype(int)

    df["gkdist_kicking"]     = df["goalkeeping_kicking"]
    df["gkdist_handling"]    = df["goalkeeping_handling"]
    df["gkdist_positioning"] = df["goalkeeping_positioning"]
    df["gkdist_reflexes"]    = df["goalkeeping_reflexes"]

    df["gksave_diving"]      = df["goalkeeping_diving"]
    df["gksave_handling"]    = df["goalkeeping_handling"]
    df["gksave_reflexes"]    = df["goalkeeping_reflexes"]
    df["gksave_speed"]       = df["goalkeeping_speed"]

    # Aggregate category scores (mean of 4 substats) — FM-card style summary
    df["score_atk"] = df[["atk_finishing","atk_shooting","atk_dribbling","atk_vision"]].mean(axis=1)
    df["score_def"] = df[["def_tackling","def_marking","def_interceptions","def_aggression"]].mean(axis=1)
    df["score_phy"] = df[["phy_pace","phy_strength","phy_stamina","phy_jumping"]].mean(axis=1)
    df["score_gk_dist"] = df[["gkdist_kicking","gkdist_handling","gkdist_positioning","gkdist_reflexes"]].mean(axis=1)
    df["score_gk_save"] = df[["gksave_diving","gksave_handling","gksave_reflexes","gksave_speed"]].mean(axis=1)

    keep = [
        "player_id","short_name","long_name",
        "nationality_id","nationality_name",
        "club_team_id","club_name","league_name","league_level",
        "age","dob","height_cm","weight_kg",
        "player_positions","preferred_foot","overall","potential",
        "pace","shooting","passing","dribbling","defending","physic",
        "is_gk",
        "atk_finishing","atk_shooting","atk_dribbling","atk_vision",
        "def_tackling","def_marking","def_interceptions","def_aggression",
        "phy_pace","phy_strength","phy_stamina","phy_jumping",
        "gkdist_kicking","gkdist_handling","gkdist_positioning","gkdist_reflexes",
        "gksave_diving","gksave_handling","gksave_reflexes","gksave_speed",
        "score_atk","score_def","score_phy","score_gk_dist","score_gk_save",
        "value_eur","wage_eur","international_reputation",
        "fifa_version","fifa_update_date",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]

    df.to_sql("players", conn, if_exists="replace", index=False)
    print(f"  saved: players ({len(df):,} rows, {len(df.columns)} cols)")


# ========================================================================
# 2) INTL_MATCHES + TEAM_ELO
# ========================================================================
def build_intl(conn):
    step("2/6  intl_matches + team_elo  (ELO from 49K matches)")

    df = pd.read_csv(RAW / "intl_results" / "results.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date","home_score","away_score"]).reset_index(drop=True)
    df["neutral"] = df["neutral"].astype(int)
    print(f"  matches: {len(df):,}")

    df.to_sql("intl_matches", conn, if_exists="replace", index=False)

    # ----- ELO computation (World Football ELO style) ---------------------
    # K weights by tournament importance
    K_WORLD_CUP = 60
    K_CONTINENT = 50      # Euro, Copa America, AFCON, etc.
    K_QUALIFIER = 40
    K_FRIENDLY  = 20

    def k_weight(tourn: str) -> int:
        t = (tourn or "").lower()
        if "world cup" in t and "qualif" not in t: return K_WORLD_CUP
        if "qualif" in t: return K_QUALIFIER
        if any(x in t for x in ["uefa euro", "copa amér", "copa america", "africa cup", "afc asian", "concacaf"]): return K_CONTINENT
        if "friendly" in t: return K_FRIENDLY
        return 30  # other tournaments

    INIT = 1500.0
    HOME_BONUS = 100.0
    elo: dict[str, float] = {}

    df = df.sort_values("date").reset_index(drop=True)
    history_rows = []

    for r in df.itertuples(index=False):
        h, a = r.home_team, r.away_team
        eh = elo.get(h, INIT)
        ea = elo.get(a, INIT)
        # apply home bonus only when not neutral
        eh_adj = eh + (HOME_BONUS if r.neutral == 0 else 0)
        # expected
        exp_h = 1.0 / (1.0 + 10 ** ((ea - eh_adj) / 400))
        # actual
        if r.home_score > r.away_score: act_h = 1.0
        elif r.home_score < r.away_score: act_h = 0.0
        else: act_h = 0.5
        # goal-difference multiplier (W. Football ELO)
        gd = abs(r.home_score - r.away_score)
        if gd <= 1: g = 1.0
        elif gd == 2: g = 1.5
        else: g = (11.0 + gd) / 8.0
        K = k_weight(r.tournament) * g
        delta = K * (act_h - exp_h)
        elo[h] = eh + delta
        elo[a] = ea - delta

    # final snapshot
    elo_df = pd.DataFrame(
        [(team, rating) for team, rating in elo.items()],
        columns=["team", "elo"],
    ).sort_values("elo", ascending=False).reset_index(drop=True)
    elo_df["rank"] = elo_df.index + 1

    elo_df.to_sql("team_elo", conn, if_exists="replace", index=False)
    print(f"  saved: team_elo ({len(elo_df):,} teams)")
    print("  Top 10 ELO:")
    print(elo_df.head(10).to_string(index=False))


# ========================================================================
# 3) WC2026 — qualified teams, schedule, venues
# ========================================================================
def build_wc2026(conn):
    step("3/6  wc2026_teams")

    teams = pd.read_csv(RAW / "wc2026" / "qual_t00.csv")
    # Clean column names
    teams = teams.rename(columns={
        "Team": "team",
        "Method of qualification": "qualification_method",
        "Date of qualification": "qualification_date",
        "Total times qualified": "total_qualifications",
        "Last time qualified": "last_qualified",
        "Current consecutive appearances": "consecutive_appearances",
        "Previous best performance": "previous_best",
    })
    # Strip trailing reference markers like "[87]" from team name
    teams["team"] = teams["team"].astype(str).str.replace(r"\[\w+\]", "", regex=True).str.strip()
    teams["is_host"] = teams["qualification_method"].astype(str).str.lower().str.contains("host").astype(int)
    teams.to_sql("wc2026_teams", conn, if_exists="replace", index=False)
    print(f"  saved: wc2026_teams ({len(teams)} teams, {teams['is_host'].sum()} hosts)")

    step("4/6  wc2026_schedule")
    sched = pd.read_csv(RAW / "wc2026" / "main_t06.csv")
    sched.columns = [c.lower() for c in sched.columns]
    sched.to_sql("wc2026_schedule", conn, if_exists="replace", index=False)
    print(f"  saved: wc2026_schedule ({len(sched)} rows)")

    step("5/6  wc2026_venues")
    venues = pd.read_csv(RAW / "wc2026" / "main_t07.csv")
    venues.columns = [c.lower() for c in venues.columns]
    venues = venues.drop(columns=[c for c in venues.columns if c == "image"], errors="ignore")
    venues.to_sql("wc2026_venues", conn, if_exists="replace", index=False)
    print(f"  saved: wc2026_venues ({len(venues)} venues)")


# ========================================================================
# 6) Sanity check — join player → wc2026_teams to see coverage
# ========================================================================
def sanity_check(conn):
    step("6/6  sanity check (player <-> WC2026 nation coverage)")

    q = """
    SELECT t.team, COUNT(p.player_id) AS player_count
    FROM wc2026_teams t
    LEFT JOIN players p ON LOWER(p.nationality_name) = LOWER(t.team)
    GROUP BY t.team
    ORDER BY player_count DESC
    """
    cov = pd.read_sql(q, conn)
    print("\n  Top 15 nations by FIFA player count (in our DB):")
    print(cov.head(15).to_string(index=False))
    print("\n  Bottom 10 (potential name mismatches):")
    print(cov.tail(10).to_string(index=False))

    missing = cov[cov["player_count"] == 0]
    if len(missing):
        print(f"\n  [WARN] {len(missing)} qualified teams have ZERO matched players — need name normalization later")
        print(missing[["team"]].to_string(index=False))


# ========================================================================
if __name__ == "__main__":
    DB_PATH.parent.mkdir(exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()  # fresh build

    conn = sqlite3.connect(DB_PATH)
    try:
        build_players(conn)
        build_intl(conn)
        build_wc2026(conn)
        sanity_check(conn)
    finally:
        conn.close()

    print(f"\n{'='*60}\nDONE -> {DB_PATH}\n{'='*60}")
