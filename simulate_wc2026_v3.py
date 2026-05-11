"""
Phase 12: WC2026 Monte Carlo with v3.0 model.

v3.0 = v2.9 (ELO + Form) + club-form ELO adjustment

For each WC team, an adjusted ELO is used:
    adjusted_elo = team_elo + (squad_club_elo - mean) * SCALE
                                                       SCALE = 0.5

This bakes "current club quality of national team players" into the ELO input
without retraining (Phase 10/11 showed adding correlated regression features
hurts).

Saves: `wc2026_predictions_v3` table.
"""
import sys, io, sqlite3, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DB_PATH = Path(__file__).parent / "data" / "soccer.db"
FEATURE_COLS = ["elo_diff", "home_advantage", "form_gd_diff", "form_cs_diff"]
CLUB_ELO_SCALE = 0.5    # how much to weight (club_elo - mean)
MIN_CLUB_COVERAGE = 5   # min matched players for club_elo to count
N_SIM = 10000
PROGRESS_EVERY = 500   # report progress every N iterations
RNG = np.random.default_rng(42)


def make_pipe():
    return Pipeline([("s", StandardScaler()),
                     ("p", PoissonRegressor(alpha=0.01, max_iter=1000))])


def train(conn):
    df = pd.read_sql("""
      SELECT m.*, f.h_form_gd, f.h_form_cs, f.a_form_gd, f.a_form_cs
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


class PredictorV3:
    """Cached predictor: precomputes lambdas for every WC team pair (48x47 = 2256)
    instead of calling sklearn predict() per match. ~1000x speedup."""
    BONUS = 100.0
    HOST_BONUS = 30.0

    def __init__(self, conn, hm, am, wc_teams: list[str]):
        self.elo = dict(zip(*pd.read_sql("SELECT team, elo FROM team_elo", conn).values.T))
        cf = pd.read_sql("SELECT * FROM current_form", conn)
        self.form_gd = dict(zip(cf["team"], cf["form_gd"].fillna(0)))
        self.form_cs = dict(zip(cf["team"], cf["form_cs"].fillna(0)))
        al = pd.read_sql("SELECT canonical, intl_name FROM nation_aliases", conn)
        self.intl_alias = dict(zip(al["canonical"], al["intl_name"]))
        self.hosts = {"United States", "Canada", "Mexico"}

        # Club ELO adjustment per WC team
        scf = pd.read_sql("SELECT * FROM squad_club_form", conn)
        valid = scf[scf["n_players_matched"] >= MIN_CLUB_COVERAGE]
        mean_club_elo = valid["squad_club_elo"].mean()
        print(f"  Mean WC squad club ELO: {mean_club_elo:.1f}  (n teams w/ enough coverage: {len(valid)})")
        self.club_adjust = {}
        for _, r in scf.iterrows():
            if pd.isna(r["squad_club_elo"]) or r["n_players_matched"] < MIN_CLUB_COVERAGE:
                self.club_adjust[r["team"]] = 0.0
            else:
                self.club_adjust[r["team"]] = (r["squad_club_elo"] - mean_club_elo) * CLUB_ELO_SCALE

        # Precompute lambdas for every pair (ta, tb) in WC teams
        print(f"  Precomputing lambdas for {len(wc_teams)*(len(wc_teams)-1)} team pairs...")
        rows = []
        pair_idx = {}
        for i, ta in enumerate(wc_teams):
            for j, tb in enumerate(wc_teams):
                if i == j: continue
                pair_idx[(ta, tb)] = len(rows)
                ea = self._adjusted_elo(ta)
                eb = self._adjusted_elo(tb)
                ha = self.HOST_BONUS if ta in self.hosts else 0.0
                gd_diff = self._lookup(ta, self.form_gd, 0.0) - self._lookup(tb, self.form_gd, 0.0)
                cs_diff = self._lookup(ta, self.form_cs, 0.0) - self._lookup(tb, self.form_cs, 0.0)
                rows.append([(ea + ha) - eb, ha, gd_diff, cs_diff])
        X = np.array(rows)
        lams_h = hm.predict(X)
        lams_a = am.predict(X)
        self.lambdas = {}
        for (ta, tb), idx in pair_idx.items():
            self.lambdas[(ta, tb)] = (float(lams_h[idx]), float(lams_a[idx]))

    def _lookup(self, team, table, default):
        if team in table: return table[team]
        return table.get(self.intl_alias.get(team, team), default)

    def _adjusted_elo(self, team):
        base = self._lookup(team, self.elo, 1500.0)
        return base + self.club_adjust.get(team, 0.0)

    def simulate_match(self, ta, tb, knockout=False):
        lh, la = self.lambdas[(ta, tb)]
        sa = int(RNG.poisson(lh)); sb = int(RNG.poisson(la))
        winner = ta if sa > sb else tb if sb > sa else None
        if knockout and winner is None:
            sa += int(RNG.poisson(lh / 3)); sb += int(RNG.poisson(la / 3))
            winner = ta if sa > sb else tb if sb > sa else None
            if winner is None:
                winner = ta if RNG.random() < 0.5 else tb
        return {"home": ta, "away": tb, "score_h": sa, "score_a": sb, "winner": winner}


def simulate_group(pred, teams):
    s = {t: {"team": t, "pts":0,"gf":0,"ga":0,"gd":0} for t in teams}
    for i in range(len(teams)):
        for j in range(i+1, len(teams)):
            ta, tb = teams[i], teams[j]
            m = pred.simulate_match(ta, tb)
            sa, sb = m["score_h"], m["score_a"]
            s[ta]["gf"] += sa; s[ta]["ga"] += sb
            s[tb]["gf"] += sb; s[tb]["ga"] += sa
            if sa > sb: s[ta]["pts"]+=3
            elif sa < sb: s[tb]["pts"]+=3
            else: s[ta]["pts"]+=1; s[tb]["pts"]+=1
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
    print(">> Train v2.9 model (still ELO+Form features)")
    hm, am, n = train(conn)
    print(f"  trained on {n:,} matches")

    print(f"\n>> Build v3 predictor (with club ELO adjustment, scale={CLUB_ELO_SCALE})")
    groups_df = pd.read_sql('SELECT team, "group" AS grp FROM wc2026_groups', conn)
    all_wc_teams = sorted(groups_df["team"].unique())
    pred = PredictorV3(conn, hm, am, all_wc_teams)

    # Show top club adjustments
    sorted_adj = sorted(pred.club_adjust.items(), key=lambda x: x[1], reverse=True)
    print("\n  Top 10 positive club ELO adjustments:")
    for t, a in sorted_adj[:10]:
        print(f"    {t:25s} {a:+7.1f}")
    print("\n  Bottom 5 club ELO adjustments:")
    for t, a in sorted_adj[-5:]:
        print(f"    {t:25s} {a:+7.1f}")

    groups = {g: list(groups_df[groups_df["grp"] == g]["team"]) for g in sorted(groups_df["grp"].unique())}

    import time
    t0 = time.time()
    print(f"\n>> Monte Carlo  N={N_SIM:,}", flush=True)
    counts = {k: Counter() for k in ["advance","r16","qf","sf","final","champion"]}
    for i in range(N_SIM):
        r = simulate_one(pred, groups)
        for t in r["advancing"]: counts["advance"][t] += 1
        for t in r["r16"]: counts["r16"][t] += 1
        for t in r["qf"]:  counts["qf"][t]  += 1
        for t in r["sf"]:  counts["sf"][t]  += 1
        for t in r["final"]: counts["final"][t] += 1
        counts["champion"][r["champion"]] += 1
        if (i+1) % PROGRESS_EVERY == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed
            eta = (N_SIM - (i+1)) / rate
            print(f"  ... {i+1:,} / {N_SIM:,}  ({elapsed:.1f}s elapsed, {rate:.0f} sims/s, ~{eta:.0f}s ETA)", flush=True)

    teams = sorted({t for tlist in groups.values() for t in tlist})
    rows = [{"team": t,
             "P(advance)":  round(counts["advance"][t]  / N_SIM, 3),
             "P(R16)":      round(counts["r16"][t]      / N_SIM, 3),
             "P(QF)":       round(counts["qf"][t]       / N_SIM, 3),
             "P(SF)":       round(counts["sf"][t]       / N_SIM, 3),
             "P(Final)":    round(counts["final"][t]    / N_SIM, 3),
             "P(Champion)": round(counts["champion"][t] / N_SIM, 4)} for t in teams]
    df = pd.DataFrame(rows).sort_values("P(Champion)", ascending=False).reset_index(drop=True)
    df.to_sql("wc2026_predictions_v3", conn, if_exists="replace", index=False)
    print(f"\nsaved: wc2026_predictions_v3 ({len(df)} teams)")

    # Comparison
    print("\n" + "="*80)
    print("v2.9 vs v3.0  (Top 20 by v3)")
    print("="*80)
    v2 = pd.read_sql('SELECT team, "P(Champion)" AS v2_win FROM wc2026_predictions_v2', conn)
    cmp = df.merge(v2, on="team").rename(columns={"P(Champion)": "v3_win"})
    cmp["delta"] = ((cmp["v3_win"] - cmp["v2_win"]) * 100).round(2)
    cmp["club_adj"] = cmp["team"].map(pred.club_adjust).round(0)
    cmp["v2_win"] = (cmp["v2_win"] * 100).round(2)
    cmp["v3_win"] = (cmp["v3_win"] * 100).round(2)
    out = cmp[["team", "club_adj", "v2_win", "v3_win", "delta"]].head(25)
    print(out.to_string(index=False))

    conn.close()


if __name__ == "__main__":
    main()
