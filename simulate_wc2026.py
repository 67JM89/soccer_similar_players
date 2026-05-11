"""
Phase 6: Monte Carlo simulation of 2026 FIFA World Cup.

WC 2026 NEW format (48 teams):
  - 12 groups of 4 (A-L), each plays 6 matches → 72 group matches
  - Top 2 from each group + 8 best 3rd-place teams → 32 advance
  - Round of 32 → R16 → QF → SF → 3rd-place + Final → 32 knockout matches
  - Total: 104 matches per tournament

Steps:
  1. Save group draw to DB (one-shot)
  2. Re-train Poisson model from match_features (without test holdout)
  3. Build helpers: simulate_match, group_standings, knockout_round
  4. Monte Carlo: simulate N=10,000 tournaments
  5. Aggregate: P(win), P(final), P(SF), P(advance from group)
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import sqlite3
import random
import re
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO

from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).parent
DB_PATH = ROOT / "data" / "soccer.db"
FEATURE_COLS = ["elo_diff", "home_advantage"]
N_SIMULATIONS = 10000
RNG = np.random.default_rng(42)


# ========================================================================
# 6.1 + 6.2  Scrape and save group draw
# ========================================================================
def scrape_groups() -> dict[str, list[str]]:
    url = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    groups: dict[str, list[str]] = {}
    for letter in "ABCDEFGHIJKL":
        h = soup.find(["h3", "h4", "h2"], id=f"Group_{letter}")
        if not h:
            h = soup.find(lambda t: t.name in ["h3","h4"] and t.get_text(strip=True) == f"Group {letter}")
        if not h: continue
        tbl = h.find_next("table", class_="wikitable")
        if not tbl: continue
        df = pd.read_html(StringIO(str(tbl)))[0]
        team_col = None
        for c in df.columns:
            if "team" in str(c).lower():
                team_col = c
                break
        if team_col is None:
            team_col = df.columns[2] if len(df.columns) > 2 else df.columns[0]
        teams = []
        for raw in df[team_col].astype(str):
            t = re.sub(r"\(H\)|\(.*?\)|\[\w+\]", "", raw).replace("vte", "").strip()
            if t and len(t) > 1 and t not in ("Source: FIFA", "Notes:"):
                teams.append(t)
        groups[letter] = teams[:4]
    return groups


def save_groups(conn, groups: dict[str, list[str]]):
    rows = []
    for grp, teams in groups.items():
        for pos, team in enumerate(teams, start=1):
            rows.append({"team": team, "group": grp, "draw_position": pos})
    df = pd.DataFrame(rows)
    df.to_sql("wc2026_groups", conn, if_exists="replace", index=False)
    return df


# ========================================================================
# Train final model on full data
# ========================================================================
def train_full_model(conn):
    df = pd.read_sql(
        "SELECT * FROM match_features WHERE date >= '2000-01-01'", conn
    )
    df = df.dropna(subset=["home_score", "away_score"])
    Xh = df[FEATURE_COLS].values
    yh = df["home_score"].values
    ya = df["away_score"].values

    def pipe():
        return Pipeline([("s", StandardScaler()), ("p", PoissonRegressor(alpha=0.01, max_iter=1000))])

    return pipe().fit(Xh, yh), pipe().fit(Xh, ya), len(df)


# ========================================================================
# Predictor with ELO lookup
# ========================================================================
class Predictor:
    BONUS = 100.0

    def __init__(self, conn, home_m, away_m):
        self.home_m = home_m
        self.away_m = away_m
        elo_df = pd.read_sql("SELECT team, elo FROM team_elo", conn)
        self.elo = dict(zip(elo_df["team"], elo_df["elo"]))
        aliases = pd.read_sql("SELECT canonical, intl_name FROM nation_aliases", conn)
        self.alias = dict(zip(aliases["canonical"], aliases["intl_name"]))
        # WC2026 hosts (host bonus applies even at "neutral" venues for them)
        self.hosts = {"United States", "Canada", "Mexico"}

    def lookup_elo(self, team: str) -> float:
        if team in self.elo: return self.elo[team]
        intl = self.alias.get(team, team)
        return self.elo.get(intl, 1500.0)

    def expected_goals(self, team_a: str, team_b: str, neutral: bool = True) -> tuple[float, float]:
        ea = self.lookup_elo(team_a)
        eb = self.lookup_elo(team_b)
        # Host advantage: hosts always get small bonus when playing at home soil
        ha = 0.0 if neutral else self.BONUS
        if team_a in self.hosts: ha += 30.0  # mild host-soil bonus
        X = np.array([[(ea + ha) - eb, ha]])
        lam_a = float(self.home_m.predict(X)[0])
        lam_b = float(self.away_m.predict(X)[0])
        return lam_a, lam_b

    def simulate_match(self, team_a: str, team_b: str, knockout: bool = False) -> dict:
        """Simulate one match. If knockout=True, resolve draws via extra time + penalties."""
        lam_a, lam_b = self.expected_goals(team_a, team_b, neutral=True)
        sa = int(RNG.poisson(lam_a))
        sb = int(RNG.poisson(lam_b))
        winner = team_a if sa > sb else team_b if sb > sa else None
        if knockout and winner is None:
            # Extra time: simulate 30 more min (1/3 of 90)
            sa_et = int(RNG.poisson(lam_a / 3))
            sb_et = int(RNG.poisson(lam_b / 3))
            sa += sa_et
            sb += sb_et
            winner = team_a if sa > sb else team_b if sb > sa else None
            if winner is None:
                # Penalty shootout: 50/50 (could weight by GK rating later)
                winner = team_a if RNG.random() < 0.5 else team_b
        return {"home": team_a, "away": team_b, "score_h": sa, "score_a": sb, "winner": winner}


# ========================================================================
# Group stage simulation
# ========================================================================
def simulate_group(predictor: Predictor, teams: list[str]) -> dict[str, dict]:
    """Returns standings: team -> {pts, gf, ga, gd, w, d, l}"""
    standings = {t: {"team": t, "pts": 0, "gf": 0, "ga": 0, "gd": 0, "w": 0, "d": 0, "l": 0} for t in teams}
    # 6 matches: round-robin
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            ta, tb = teams[i], teams[j]
            m = predictor.simulate_match(ta, tb, knockout=False)
            sa, sb = m["score_h"], m["score_a"]
            standings[ta]["gf"] += sa; standings[ta]["ga"] += sb
            standings[tb]["gf"] += sb; standings[tb]["ga"] += sa
            if sa > sb:   standings[ta]["pts"] += 3; standings[ta]["w"] += 1; standings[tb]["l"] += 1
            elif sa < sb: standings[tb]["pts"] += 3; standings[tb]["w"] += 1; standings[ta]["l"] += 1
            else:         standings[ta]["pts"] += 1; standings[tb]["pts"] += 1; standings[ta]["d"] += 1; standings[tb]["d"] += 1
    for s in standings.values():
        s["gd"] = s["gf"] - s["ga"]
    return standings


def rank_group(standings: dict) -> list[dict]:
    """Sort by pts → gd → gf, with random tiebreak last."""
    items = list(standings.values())
    for s in items: s["_rand"] = RNG.random()
    items.sort(key=lambda s: (-s["pts"], -s["gd"], -s["gf"], s["_rand"]))
    return items


# ========================================================================
# 32-team knockout (NEW WC2026 format)
# ========================================================================
def select_8_best_thirds(thirds: list[dict]) -> list[dict]:
    """Pick 8 best third-placed teams by pts → gd → gf."""
    for s in thirds: s["_rand"] = RNG.random()
    thirds.sort(key=lambda s: (-s["pts"], -s["gd"], -s["gf"], s["_rand"]))
    return thirds[:8]


def simulate_one_tournament(predictor: Predictor, groups: dict[str, list[str]]) -> dict:
    # ----- Group stage
    g_stand: dict[str, list[dict]] = {}
    firsts, seconds, thirds = [], [], []
    for g, teams in groups.items():
        st = simulate_group(predictor, teams)
        ranked = rank_group(st)
        for s in ranked:
            s["group"] = g
        firsts.append(ranked[0])
        seconds.append(ranked[1])
        thirds.append(ranked[2])
        g_stand[g] = ranked

    best_thirds = select_8_best_thirds(thirds)
    # 32 advancing teams
    advancing = firsts + seconds + best_thirds
    advance_set = {s["team"] for s in advancing}

    # ----- Round of 32 — simplified bracket: pair 1st-vs-3rd, 2nd-vs-2nd ...
    # FIFA's exact bracket logic uses position-based seeding; for MVP use random pairing
    bracket = [s["team"] for s in advancing]
    RNG.shuffle(bracket)

    rounds_results = {}

    def play_round(name: str, teams: list[str]) -> list[str]:
        winners = []
        for i in range(0, len(teams), 2):
            m = predictor.simulate_match(teams[i], teams[i + 1], knockout=True)
            winners.append(m["winner"])
        rounds_results[name] = winners
        return winners

    r16 = play_round("R32", bracket)            # 16 winners
    qf  = play_round("R16", r16)                # 8 winners
    sf  = play_round("QF", qf)                  # 4 winners
    final2 = play_round("SF", sf)               # 2 winners → final

    # 3rd place playoff
    sf_losers = [t for t in sf if t not in final2]
    third = predictor.simulate_match(sf_losers[0], sf_losers[1], knockout=True)["winner"]

    # Final
    fin = predictor.simulate_match(final2[0], final2[1], knockout=True)
    champion = fin["winner"]
    runner_up = final2[1] if final2[0] == champion else final2[0]

    return {
        "groups": g_stand,
        "advancing": list(advance_set),
        "r16": r16,
        "qf":  qf,
        "sf":  sf,
        "final": final2,
        "champion": champion,
        "runner_up": runner_up,
        "third": third,
    }


# ========================================================================
# Monte Carlo
# ========================================================================
def monte_carlo(predictor: Predictor, groups: dict[str, list[str]], n: int):
    counts = {
        "advance":   Counter(),
        "r16":       Counter(),
        "qf":        Counter(),
        "sf":        Counter(),
        "final":     Counter(),
        "champion":  Counter(),
        "third":     Counter(),
    }
    all_teams = [t for tlist in groups.values() for t in tlist]
    for t in all_teams:
        for k in counts:
            counts[k][t] = 0

    for i in range(n):
        result = simulate_one_tournament(predictor, groups)
        for t in result["advancing"]: counts["advance"][t] += 1
        for t in result["r16"]: counts["r16"][t] += 1
        for t in result["qf"]:  counts["qf"][t]  += 1
        for t in result["sf"]:  counts["sf"][t]  += 1
        for t in result["final"]: counts["final"][t] += 1
        counts["champion"][result["champion"]] += 1
        counts["third"][result["third"]] += 1
        if (i + 1) % 1000 == 0:
            print(f"  ... {i+1:,} / {n:,}")

    return counts


def report(counts: Counter, n: int) -> pd.DataFrame:
    teams = sorted(counts["champion"].keys())
    rows = []
    for t in teams:
        rows.append({
            "team": t,
            "P(advance)":  round(counts["advance"][t]  / n, 3),
            "P(R16)":      round(counts["r16"][t]      / n, 3),
            "P(QF)":       round(counts["qf"][t]       / n, 3),
            "P(SF)":       round(counts["sf"][t]       / n, 3),
            "P(Final)":    round(counts["final"][t]    / n, 3),
            "P(Champion)": round(counts["champion"][t] / n, 4),
        })
    return pd.DataFrame(rows).sort_values("P(Champion)", ascending=False).reset_index(drop=True)


# ========================================================================
def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        print(">> 6.1/6.2  Scrape + save group draw")
        groups = scrape_groups()
        save_groups(conn, groups)
        print(f"  saved: wc2026_groups ({sum(len(v) for v in groups.values())} teams in {len(groups)} groups)")

        print("\n>> 6.3  Train final model")
        home_m, away_m, n_train = train_full_model(conn)
        print(f"  trained on {n_train:,} matches")

        predictor = Predictor(conn, home_m, away_m)

        print(f"\n>> 6.4  Monte Carlo  (N = {N_SIMULATIONS:,})")
        counts = monte_carlo(predictor, groups, N_SIMULATIONS)

        print(f"\n>> 6.5  Aggregate results")
        df = report(counts, N_SIMULATIONS)
        df.to_sql("wc2026_predictions", conn, if_exists="replace", index=False)
        print(f"  saved: wc2026_predictions ({len(df)} teams)")

        print("\n" + "=" * 70)
        print("WC2026 PREDICTION — Top 15 by Championship Probability")
        print("=" * 70)
        print(df.head(15).to_string(index=False))

        print("\n" + "=" * 70)
        print("BOTTOM 10")
        print("=" * 70)
        print(df.tail(10).to_string(index=False))

    finally:
        conn.close()


if __name__ == "__main__":
    main()
