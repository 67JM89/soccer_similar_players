"""
Phase 10: Squad strength feature using point-in-time FIFA editions.

For each match year Y, use FIFA edition closest to that year:
   year 2014 -> FIFA 15  (released 2014)
   year 2018 -> FIFA 19
   year 2022 -> FIFA 22

Squad strength = mean of top-25 players' overall rating, per nationality.

Add new feature:  squad_overall_diff = home_squad - away_squad
"""
import sys, io, sqlite3, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, brier_score_loss

ROOT = Path(__file__).parent
DB_PATH = ROOT / "data" / "soccer.db"
FIFA_DIR = ROOT / "data" / "raw" / "fifa22"

# Map a match year to the relevant FIFA edition file (released previous fall)
# FIFA 15 covers ~Aug 2014 - Jul 2015, etc.
def fifa_file_for(year: int) -> Path:
    # FIFA edition: max(15, min(22, year-13))  → 2014→15, 2017→17, 2022→22
    edition = max(15, min(22, year - 13))
    return FIFA_DIR / f"players_{edition:02d}.csv"


# Name alias mapping (FIFA -> intl_matches)
FIFA_TO_INTL = {
    "Korea Republic": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
    "Curacao": "Curaçao",
    "Cape Verde Islands": "Cape Verde",
    "China PR": "China PR",  # same
}


def build_squad_strength(year: int) -> dict[str, float]:
    """Returns {nation_name: mean of top-25 overall} using FIFA edition for that year."""
    fp = fifa_file_for(year)
    if not fp.exists():
        return {}
    df = pd.read_csv(fp, low_memory=False, usecols=["nationality", "overall"]) \
        if "nationality" in pd.read_csv(fp, nrows=0).columns \
        else pd.read_csv(fp, low_memory=False, usecols=["nationality_name", "overall"])
    nat_col = "nationality" if "nationality" in df.columns else "nationality_name"

    result = {}
    for nation, sub in df.groupby(nat_col):
        top25 = sub.nlargest(25, "overall")
        result[nation] = float(top25["overall"].mean())

    # Apply aliases — map FIFA name to intl_matches name
    aliased = {}
    for fifa_name, score in result.items():
        intl_name = FIFA_TO_INTL.get(fifa_name, fifa_name)
        aliased[intl_name] = score
    return aliased


def make_pipe():
    return Pipeline([("s", StandardScaler()),
                     ("p", PoissonRegressor(alpha=0.01, max_iter=1000))])


def load(conn):
    df = pd.read_sql("""
      SELECT m.*, f.h_form_gd, f.h_form_cs, f.a_form_gd, f.a_form_cs
      FROM match_features m
      JOIN match_form f ON f.date=m.date AND f.home_team=m.home_team AND f.away_team=m.away_team
    """, conn)
    df["date"] = pd.to_datetime(df["date"])
    df["form_gd_diff"] = df["h_form_gd"] - df["a_form_gd"]
    df["form_cs_diff"] = df["h_form_cs"] - df["a_form_cs"]
    df["t_low"] = df["tournament"].str.lower()
    df["year"] = df["date"].dt.year
    return df


def attach_squad(df: pd.DataFrame) -> pd.DataFrame:
    """Add squad_h, squad_a, squad_diff using point-in-time FIFA editions."""
    years_in_data = sorted(df["year"].unique())
    squad_cache: dict[int, dict[str, float]] = {}
    for y in years_in_data:
        edition_year = max(2014, min(2022, y))
        if edition_year not in squad_cache:
            squad_cache[edition_year] = build_squad_strength(edition_year)

    def lookup(team: str, year: int):
        edition_year = max(2014, min(2022, year))
        return squad_cache[edition_year].get(team, np.nan)

    df["squad_h"] = df.apply(lambda r: lookup(r["home_team"], r["year"]), axis=1)
    df["squad_a"] = df.apply(lambda r: lookup(r["away_team"], r["year"]), axis=1)
    df["squad_diff"] = df["squad_h"] - df["squad_a"]
    return df


def probs(lh, la, K=10):
    ph = poisson.pmf(np.arange(K+1), lh); pa = poisson.pmf(np.arange(K+1), la)
    g = np.outer(ph, pa)
    return float(np.tril(g,-1).sum()), float(np.diag(g).sum()), float(np.triu(g,1).sum())


def eval_set(test, feats, hm, am):
    test = test.dropna(subset=feats).copy()
    X = test[feats].values
    p = np.array([probs(h, a) for h, a in zip(hm.predict(X), am.predict(X))])
    test["p_h"] = p[:,0]; test["p_d"] = p[:,1]; test["p_a"] = p[:,2]
    test["pred"] = np.where(test["p_h"] > np.maximum(test["p_d"], test["p_a"]), 1,
                    np.where(test["p_a"] > np.maximum(test["p_h"], test["p_d"]), -1, 0))
    y = test["result"].values
    return {
        "n": len(test),
        "acc": round(accuracy_score(y, test["pred"]), 4),
        "brier": round((brier_score_loss((y==1).astype(int), test["p_h"]) +
                        brier_score_loss((y==0).astype(int), test["p_d"]) +
                        brier_score_loss((y==-1).astype(int), test["p_a"])) / 3, 4),
    }


def run(df, test_pool, years, feats):
    accs, briers = [], []
    for y in years:
        train = df[(df["date"] >= "2000-01-01") & (df["date"] < f"{y}-01-01")]
        test = test_pool[test_pool["date"].dt.year == y]
        if len(train) < 100 or len(test) < 5:
            accs.append(np.nan); briers.append(np.nan); continue
        train_x = train.dropna(subset=feats)
        hm = make_pipe().fit(train_x[feats], train_x["home_score"])
        am = make_pipe().fit(train_x[feats], train_x["away_score"])
        r = eval_set(test, feats, hm, am)
        accs.append(r["acc"]); briers.append(r["brier"])
    return float(np.nanmean(accs)), float(np.nanmean(briers)), accs


def main():
    conn = sqlite3.connect(DB_PATH)
    print(">> Load + attach point-in-time squad strength")
    df = load(conn)
    df = attach_squad(df)
    print(f"  matches with squad_diff: {df['squad_diff'].notna().sum()} / {len(df)}")
    print(f"  unique years: {df['year'].nunique()}")

    df = df.dropna(subset=["form_gd_diff", "form_cs_diff"])

    big = ["fifa world cup", "uefa euro", "copa américa", "copa america",
           "african cup of nations", "afc asian cup"]
    test_pool = df[df["t_low"].isin(big) & (df["date"] >= "2014-01-01")]
    print(f"\n  test pool: {len(test_pool)}")
    print(f"  test pool with squad_diff: {test_pool['squad_diff'].notna().sum()}")
    years = [2014, 2016, 2018, 2020, 2022, 2024]

    base = ["elo_diff", "home_advantage", "form_gd_diff", "form_cs_diff"]
    variants = {
        "v2.9 (base)":      base,
        "v6a +squad_diff":  base + ["squad_diff"],
        "v6b +squad_h+a":   base + ["squad_h", "squad_a"],
    }

    print(f"\n{'variant':25s}  " + "  ".join(f"{y:6d}" for y in years) + f"   {'mean':>6s}  {'brier':>7s}")
    print("-" * 100)
    for label, feats in variants.items():
        m_acc, m_b, accs = run(df, test_pool, years, feats)
        accs_str = "  ".join(f"{a:.3f}" if not np.isnan(a) else " --- " for a in accs)
        print(f"{label:25s}  {accs_str}   {m_acc:.3f}   {m_b:.4f}")

    # Diagnostic: show squad coverage for major teams
    print("\nSquad coverage spot-check (2022 edition):")
    s22 = build_squad_strength(2022)
    for t in ["Brazil","Argentina","Spain","France","Germany","Senegal","Morocco","Japan","USA","Mexico"]:
        print(f"  {t}: {s22.get(t, 'MISSING'):.2f}" if isinstance(s22.get(t), float) else f"  {t}: MISSING")
    conn.close()


if __name__ == "__main__":
    main()
