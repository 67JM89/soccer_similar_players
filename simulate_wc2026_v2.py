"""
Phase 7c: WC2026 Monte Carlo with v2.9 model (ELO + form_gd_diff + form_cs_diff).

Differences from simulate_wc2026.py (v1):
  - Uses 5-feature Poisson model
  - Predictor pulls each team's current form from `current_form` table
  - Saves results to `wc2026_predictions_v2`
"""
import sys, io, sqlite3, warnings, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DB_PATH = Path(__file__).parent / "data" / "soccer.db"
FEATURE_COLS = ["elo_diff", "home_advantage", "form_gd_diff", "form_cs_diff"]
N_SIM = 10000
RNG = np.random.default_rng(42)


def make_pipe():
    return Pipeline([("s", StandardScaler()),
                     ("p", PoissonRegressor(alpha=0.01, max_iter=1000))])


def train(conn):
    df = pd.read_sql("""
      SELECT m.*, f.h_form_ppg, f.h_form_gd, f.h_form_cs,
                  f.a_form_ppg, f.a_form_gd, f.a_form_cs
      FROM match_features m
      JOIN match_form f ON f.date=m.date AND f.home_team=m.home_team AND f.away_team=m.away_team
    """, conn)
    df["date"] = pd.to_datetime(df["date"])
    df["form_gd_diff"] = df["h_form_gd"] - df["a_form_gd"]
    df["form_cs_diff"] = df["h_form_cs"] - df["a_form_cs"]
    df = df[df["date"] >= "2000-01-01"].dropna(subset=FEATURE_COLS)
    Xh = df[FEATURE_COLS].values
    hm = make_pipe().fit(Xh, df["home_score"].values)
    am = make_pipe().fit(Xh, df["away_score"].values)
    return hm, am, len(df)


class Predictor:
    BONUS = 100.0
    HOST_BONUS = 30.0

    def __init__(self, conn, hm, am):
        self.hm, self.am = hm, am
        self.elo = dict(zip(*pd.read_sql("SELECT team, elo FROM team_elo", conn).values.T))
        # current form lookup
        cf = pd.read_sql("SELECT * FROM current_form", conn)
        self.form_gd = dict(zip(cf["team"], cf["form_gd"].fillna(0)))
        self.form_cs = dict(zip(cf["team"], cf["form_cs"].fillna(0)))
        # alias
        al = pd.read_sql("SELECT canonical, intl_name, fifa_name FROM nation_aliases", conn)
        self.intl_alias = dict(zip(al["canonical"], al["intl_name"]))
        self.hosts = {"United States", "Canada", "Mexico"}

    def _lookup(self, team, table, default):
        if team in table: return table[team]
        return table.get(self.intl_alias.get(team, team), default)

    def features(self, ta, tb, neutral=True):
        ea = self._lookup(ta, self.elo, 1500.0)
        eb = self._lookup(tb, self.elo, 1500.0)
        ha = 0.0 if neutral else self.BONUS
        if ta in self.hosts: ha += self.HOST_BONUS
        gd_diff = self._lookup(ta, self.form_gd, 0.0) - self._lookup(tb, self.form_gd, 0.0)
        cs_diff = self._lookup(ta, self.form_cs, 0.0) - self._lookup(tb, self.form_cs, 0.0)
        return np.array([[(ea + ha) - eb, ha, gd_diff, cs_diff]])

    def expected_goals(self, ta, tb):
        X = self.features(ta, tb, neutral=True)
        return float(self.hm.predict(X)[0]), float(self.am.predict(X)[0])

    def simulate_match(self, ta, tb, knockout=False):
        lh, la = self.expected_goals(ta, tb)
        sa = int(RNG.poisson(lh)); sb = int(RNG.poisson(la))
        winner = ta if sa > sb else tb if sb > sa else None
        if knockout and winner is None:
            sa += int(RNG.poisson(lh / 3)); sb += int(RNG.poisson(la / 3))
            winner = ta if sa > sb else tb if sb > sa else None
            if winner is None:
                winner = ta if RNG.random() < 0.5 else tb
        return {"home": ta, "away": tb, "score_h": sa, "score_a": sb, "winner": winner}


# Group/knockout simulation logic copied from v1
def simulate_group(pred, teams):
    s = {t: {"team": t, "pts":0,"gf":0,"ga":0,"gd":0,"w":0,"d":0,"l":0} for t in teams}
    for i in range(len(teams)):
        for j in range(i+1, len(teams)):
            ta, tb = teams[i], teams[j]
            m = pred.simulate_match(ta, tb)
            sa, sb = m["score_h"], m["score_a"]
            s[ta]["gf"] += sa; s[ta]["ga"] += sb
            s[tb]["gf"] += sb; s[tb]["ga"] += sa
            if sa > sb: s[ta]["pts"]+=3; s[ta]["w"]+=1; s[tb]["l"]+=1
            elif sa < sb: s[tb]["pts"]+=3; s[tb]["w"]+=1; s[ta]["l"]+=1
            else: s[ta]["pts"]+=1; s[tb]["pts"]+=1; s[ta]["d"]+=1; s[tb]["d"]+=1
    for v in s.values(): v["gd"] = v["gf"] - v["ga"]
    return s

def rank_group(st):
    items = list(st.values())
    for v in items: v["_r"] = RNG.random()
    items.sort(key=lambda v: (-v["pts"], -v["gd"], -v["gf"], v["_r"]))
    return items

def best_thirds(thirds):
    for v in thirds: v["_r"] = RNG.random()
    thirds.sort(key=lambda v: (-v["pts"], -v["gd"], -v["gf"], v["_r"]))
    return thirds[:8]

def simulate_one(pred, groups):
    firsts, seconds, thirds = [], [], []
    for g, teams in groups.items():
        ranked = rank_group(simulate_group(pred, teams))
        for r in ranked: r["group"] = g
        firsts.append(ranked[0]); seconds.append(ranked[1]); thirds.append(ranked[2])
    advancing = firsts + seconds + best_thirds(thirds)
    bracket = [s["team"] for s in advancing]; RNG.shuffle(bracket)
    def play(round_teams):
        return [pred.simulate_match(round_teams[i], round_teams[i+1], knockout=True)["winner"]
                for i in range(0, len(round_teams), 2)]
    r16 = play(bracket); qf = play(r16); sf = play(qf); fin2 = play(sf)
    fin = pred.simulate_match(fin2[0], fin2[1], knockout=True)
    return {"advancing": [s["team"] for s in advancing], "r16": r16, "qf": qf,
            "sf": sf, "final": fin2, "champion": fin["winner"]}


def main():
    conn = sqlite3.connect(DB_PATH)
    print(">> Train v2.9 final model")
    hm, am, n = train(conn)
    print(f"  trained on {n:,} matches  (features: {FEATURE_COLS})")
    h_pois = hm.named_steps["p"]
    print(f"  home coefs (scaled): {dict(zip(FEATURE_COLS, h_pois.coef_.round(4)))}")

    pred = Predictor(conn, hm, am)
    groups_df = pd.read_sql('SELECT team, "group" AS grp FROM wc2026_groups', conn)
    groups = {g: list(groups_df[groups_df["grp"] == g]["team"]) for g in sorted(groups_df["grp"].unique())}

    print(f"\n>> Monte Carlo  N={N_SIM:,}")
    counts = {k: Counter() for k in ["advance","r16","qf","sf","final","champion"]}
    for i in range(N_SIM):
        r = simulate_one(pred, groups)
        for t in r["advancing"]: counts["advance"][t] += 1
        for t in r["r16"]: counts["r16"][t] += 1
        for t in r["qf"]:  counts["qf"][t]  += 1
        for t in r["sf"]:  counts["sf"][t]  += 1
        for t in r["final"]: counts["final"][t] += 1
        counts["champion"][r["champion"]] += 1
        if (i+1) % 1000 == 0: print(f"  ... {i+1:,} / {N_SIM:,}")

    teams = sorted({t for tlist in groups.values() for t in tlist})
    rows = [{"team": t,
             "P(advance)":  round(counts["advance"][t]  / N_SIM, 3),
             "P(R16)":      round(counts["r16"][t]      / N_SIM, 3),
             "P(QF)":       round(counts["qf"][t]       / N_SIM, 3),
             "P(SF)":       round(counts["sf"][t]       / N_SIM, 3),
             "P(Final)":    round(counts["final"][t]    / N_SIM, 3),
             "P(Champion)": round(counts["champion"][t] / N_SIM, 4)} for t in teams]
    df = pd.DataFrame(rows).sort_values("P(Champion)", ascending=False).reset_index(drop=True)
    df.to_sql("wc2026_predictions_v2", conn, if_exists="replace", index=False)
    print(f"\nsaved: wc2026_predictions_v2 ({len(df)} teams)")

    print("\n" + "="*70)
    print("WC2026 v2.9 (ELO + Form) — Top 15")
    print("="*70)
    print(df.head(15).to_string(index=False))

    # Compare v1 vs v2 (top 15)
    v1 = pd.read_sql("SELECT team, \"P(Champion)\" FROM wc2026_predictions ORDER BY \"P(Champion)\" DESC", conn).head(15)
    v1 = v1.rename(columns={"P(Champion)": "v1_champ"})
    cmp = df.head(15).merge(v1, on="team", how="left").rename(columns={"P(Champion)": "v2_champ"})
    cmp["delta"] = (cmp["v2_champ"] - cmp["v1_champ"]).round(4)
    print("\nComparison v1 vs v2.9 (top 15 by v2):")
    print(cmp[["team","v1_champ","v2_champ","delta"]].to_string(index=False))

    conn.close()


if __name__ == "__main__":
    main()
