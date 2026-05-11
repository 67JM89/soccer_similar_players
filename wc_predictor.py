"""
Reusable predictor module — v3.5 final model.

v3.5 = ELO + Form (gd, cs) + Club ELO adjustment + Dixon-Coles τ correction

Public API:
    predictor = Predictor()
    result = predictor.predict_match("Spain", "Brazil")
    # → {p_a_win, p_draw, p_b_win, expected_a_goals, expected_b_goals, score_grid}
"""
import sqlite3
import warnings
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

DB_PATH = Path(__file__).parent / "data" / "soccer.db"
FEATURE_COLS = ["elo_diff", "home_advantage", "form_gd_diff", "form_cs_diff"]

# Hyperparameters (from backtest tuning)
DC_RHO = -0.05               # Dixon-Coles correction
CLUB_ELO_SCALE = 0.5         # squad club ELO weight
MIN_CLUB_COVERAGE = 5
HOME_BONUS = 100.0
HOST_BONUS = 30.0
HOSTS_2026 = {"United States", "Canada", "Mexico"}


def _make_pipe():
    return Pipeline([("s", StandardScaler()),
                     ("p", PoissonRegressor(alpha=0.01, max_iter=1000))])


class Predictor:
    """v3.5 predictor: train once on full data, predict any team pair instantly."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self._train()
        self._build_lookups()

    def _train(self):
        conn = sqlite3.connect(self.db_path)
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
        self.hm = _make_pipe().fit(Xh, df["home_score"].values)
        self.am = _make_pipe().fit(Xh, df["away_score"].values)
        self.n_train = len(df)
        conn.close()

    def _build_lookups(self):
        conn = sqlite3.connect(self.db_path)
        elo_df = pd.read_sql("SELECT team, elo FROM team_elo", conn)
        self.elo = dict(zip(elo_df["team"], elo_df["elo"]))

        cf = pd.read_sql("SELECT * FROM current_form", conn)
        self.form_gd = dict(zip(cf["team"], cf["form_gd"].fillna(0)))
        self.form_cs = dict(zip(cf["team"], cf["form_cs"].fillna(0)))

        al = pd.read_sql("SELECT canonical, intl_name FROM nation_aliases", conn)
        self.alias = dict(zip(al["canonical"], al["intl_name"]))

        try:
            scf = pd.read_sql("SELECT * FROM squad_club_form", conn)
            valid = scf[scf["n_players_matched"] >= MIN_CLUB_COVERAGE]
            self.mean_club_elo = float(valid["squad_club_elo"].mean())
            self.club_adjust = {}
            for _, r in scf.iterrows():
                if pd.isna(r["squad_club_elo"]) or r["n_players_matched"] < MIN_CLUB_COVERAGE:
                    self.club_adjust[r["team"]] = 0.0
                else:
                    self.club_adjust[r["team"]] = (r["squad_club_elo"] - self.mean_club_elo) * CLUB_ELO_SCALE
        except pd.errors.DatabaseError:
            self.club_adjust = {}
            self.mean_club_elo = 1500.0
        conn.close()

    def _lookup(self, team: str, table: dict, default: float) -> float:
        if team in table: return table[team]
        return table.get(self.alias.get(team, team), default)

    def adjusted_elo(self, team: str) -> float:
        return self._lookup(team, self.elo, 1500.0) + self.club_adjust.get(team, 0.0)

    def expected_goals(self, team_a: str, team_b: str,
                       neutral: bool = True,
                       apply_host_bonus: bool = True) -> tuple[float, float]:
        ea = self.adjusted_elo(team_a)
        eb = self.adjusted_elo(team_b)
        ha = 0.0 if neutral else HOME_BONUS
        if apply_host_bonus and team_a in HOSTS_2026: ha += HOST_BONUS
        gd_diff = self._lookup(team_a, self.form_gd, 0.0) - self._lookup(team_b, self.form_gd, 0.0)
        cs_diff = self._lookup(team_a, self.form_cs, 0.0) - self._lookup(team_b, self.form_cs, 0.0)
        X = np.array([[(ea + ha) - eb, ha, gd_diff, cs_diff]])
        return float(self.hm.predict(X)[0]), float(self.am.predict(X)[0])

    def predict_match(self, team_a: str, team_b: str,
                       neutral: bool = True, apply_host_bonus: bool = True) -> dict:
        """Returns prediction with Dixon-Coles correction applied."""
        lam_a, lam_b = self.expected_goals(team_a, team_b, neutral, apply_host_bonus)

        K = 8
        ph = poisson.pmf(np.arange(K + 1), lam_a)
        pb = poisson.pmf(np.arange(K + 1), lam_b)
        grid = np.outer(ph, pb)
        # Dixon-Coles τ correction on low-score cells
        grid[0, 0] *= (1 - lam_a * lam_b * DC_RHO)
        grid[1, 0] *= (1 + lam_b * DC_RHO)
        grid[0, 1] *= (1 + lam_a * DC_RHO)
        grid[1, 1] *= (1 - DC_RHO)

        p_a_win = float(np.tril(grid, -1).sum())
        p_draw  = float(np.diag(grid).sum())
        p_b_win = float(np.triu(grid,  1).sum())
        # normalize (small DC mass shift)
        total = p_a_win + p_draw + p_b_win
        p_a_win, p_draw, p_b_win = p_a_win/total, p_draw/total, p_b_win/total

        # Most likely scoreline
        most_likely_idx = np.unravel_index(np.argmax(grid), grid.shape)

        # Top 5 likely scorelines
        flat = grid.flatten()
        top_idx = np.argsort(flat)[::-1][:5]
        top_scores = []
        for idx in top_idx:
            a_s, b_s = np.unravel_index(idx, grid.shape)
            top_scores.append({"score": f"{a_s}-{b_s}", "prob": float(flat[idx] / total)})

        return {
            "team_a": team_a, "team_b": team_b, "neutral": neutral,
            "elo_a": round(self.adjusted_elo(team_a), 1),
            "elo_b": round(self.adjusted_elo(team_b), 1),
            "expected_a_goals": round(lam_a, 2),
            "expected_b_goals": round(lam_b, 2),
            "p_a_win": round(p_a_win, 4),
            "p_draw":  round(p_draw,  4),
            "p_b_win": round(p_b_win, 4),
            "most_likely_score": f"{most_likely_idx[0]}-{most_likely_idx[1]}",
            "top_scores": top_scores,
            "score_grid": grid / total,
        }

    def list_teams(self, wc2026_only: bool = False) -> list[str]:
        conn = sqlite3.connect(self.db_path)
        if wc2026_only:
            df = pd.read_sql("SELECT team FROM wc2026_teams ORDER BY team", conn)
        else:
            df = pd.read_sql("SELECT DISTINCT team FROM team_elo ORDER BY team", conn)
        conn.close()
        return df["team"].tolist()


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    p = Predictor()
    print(f"Trained on {p.n_train:,} matches.")
    print(f"Has club adjustment for {sum(1 for v in p.club_adjust.values() if v != 0)} teams.\n")

    for ta, tb in [("Spain", "Argentina"), ("France", "Brazil"),
                   ("Germany", "Mexico"), ("Japan", "Morocco")]:
        r = p.predict_match(ta, tb, neutral=True)
        print(f"{ta} vs {tb}: xG={r['expected_a_goals']}-{r['expected_b_goals']}  "
              f"P({ta})={r['p_a_win']*100:.1f}% / D={r['p_draw']*100:.1f}% / "
              f"P({tb})={r['p_b_win']*100:.1f}%   most likely: {r['most_likely_score']}")
