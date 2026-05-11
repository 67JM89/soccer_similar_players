"""
Phase 11: Host / Continental advantage features.

New features:
  - h_continent_match : 1 if home_team's continent == match country's continent
  - a_continent_match : 1 if away_team's continent == match country's continent
  - continent_diff    : h_continent_match - a_continent_match  (range -1, 0, +1)
  - is_host_h         : 1 if home_team == match country
  - is_host_a         : 1 if away_team == match country

Test variants (with v2.9 base features):
  - v5a: + continent_diff only
  - v5b: + is_host_h + is_host_a
  - v5c: + continent_diff + is_host_h
  - v5d: full (all 5 new features)
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

DB_PATH = Path(__file__).parent / "data" / "soccer.db"

# Continent mapping based on FIFA confederations
CONTINENT = {}
def _add(continent, names):
    for n in names: CONTINENT[n] = continent

_add("UEFA", [
    "Albania","Andorra","Armenia","Austria","Azerbaijan","Belarus","Belgium",
    "Bosnia and Herzegovina","Bulgaria","Croatia","Cyprus","Czech Republic",
    "Czechoslovakia","Denmark","England","Estonia","Faroe Islands","Finland",
    "France","Georgia","Germany","German DR","East Germany","Gibraltar","Greece",
    "Hungary","Iceland","Republic of Ireland","Ireland","Israel","Italy",
    "Kazakhstan","Kosovo","Latvia","Liechtenstein","Lithuania","Luxembourg",
    "Malta","Moldova","Monaco","Montenegro","Netherlands","North Macedonia",
    "Macedonia","Northern Ireland","Norway","Poland","Portugal","Romania",
    "Russia","San Marino","Scotland","Serbia","Serbia and Montenegro","Slovakia",
    "Slovenia","Spain","Sweden","Switzerland","Turkey","Türkiye","Ukraine",
    "Soviet Union","Wales","Yugoslavia",
])
_add("CONMEBOL", [
    "Argentina","Bolivia","Brazil","Chile","Colombia","Ecuador",
    "Paraguay","Peru","Uruguay","Venezuela",
])
_add("CONCACAF", [
    "Antigua and Barbuda","Aruba","Bahamas","Barbados","Belize","Bermuda",
    "British Virgin Islands","Canada","Cayman Islands","Costa Rica","Cuba",
    "Curaçao","Curacao","Dominica","Dominican Republic","El Salvador",
    "Grenada","Guadeloupe","Guatemala","Guyana","Haiti","Honduras","Jamaica",
    "Martinique","Mexico","Montserrat","Netherlands Antilles","Nicaragua",
    "Panama","Puerto Rico","Saint Kitts and Nevis","Saint Lucia",
    "Saint Martin","Saint Vincent and the Grenadines","Sint Maarten",
    "Suriname","Trinidad and Tobago","Turks and Caicos Islands",
    "United States","US Virgin Islands","French Guiana",
])
_add("AFC", [
    "Afghanistan","Australia","Bahrain","Bangladesh","Bhutan","Brunei",
    "Cambodia","China PR","China","Chinese Taipei","Taiwan","East Timor",
    "Timor-Leste","Guam","Hong Kong","India","Indonesia","Iran","Iraq",
    "Japan","Jordan","North Korea","Korea DPR","South Korea","Korea Republic",
    "Kuwait","Kyrgyzstan","Laos","Lebanon","Macau","Malaysia","Maldives",
    "Mongolia","Myanmar","Nepal","Northern Mariana Islands","Oman","Pakistan",
    "Palestine","Philippines","Qatar","Saudi Arabia","Singapore","Sri Lanka",
    "Syria","Tajikistan","Thailand","Turkmenistan","United Arab Emirates",
    "Uzbekistan","Vietnam","Yemen",
])
_add("CAF", [
    "Algeria","Angola","Benin","Botswana","Burkina Faso","Burundi","Cameroon",
    "Cape Verde","Cape Verde Islands","Central African Republic","Chad",
    "Comoros","Congo","Congo DR","DR Congo","Republic of Congo",
    "Côte d'Ivoire","Ivory Coast","Djibouti","Egypt","Equatorial Guinea",
    "Eritrea","Eswatini","Swaziland","Ethiopia","Gabon","Gambia","Ghana",
    "Guinea","Guinea-Bissau","Kenya","Lesotho","Liberia","Libya","Madagascar",
    "Malawi","Mali","Mauritania","Mauritius","Morocco","Mozambique","Namibia",
    "Niger","Nigeria","Rwanda","São Tomé and Príncipe","Senegal","Seychelles",
    "Sierra Leone","Somalia","South Africa","South Sudan","Sudan","Tanzania",
    "Togo","Tunisia","Uganda","Zambia","Zanzibar","Zimbabwe","Réunion",
    "Western Sahara","Mayotte",
])
_add("OFC", [
    "American Samoa","Cook Islands","Fiji","Kiribati","New Caledonia",
    "New Zealand","Niue","Papua New Guinea","Samoa","Solomon Islands",
    "Tahiti","Tonga","Tuvalu","Vanuatu",
])


def continent_of(name: str) -> str:
    if name in CONTINENT: return CONTINENT[name]
    # alias fallbacks
    alias = {
        "United States Virgin Islands": "CONCACAF",
        "Czech Republic": "UEFA",
    }
    return alias.get(name, "OTHER")


def make_pipe():
    return Pipeline([("s", StandardScaler()),
                     ("p", PoissonRegressor(alpha=0.01, max_iter=1000))])


def load(conn):
    df = pd.read_sql("""
      SELECT m.*, f.h_form_gd, f.h_form_cs, f.a_form_gd, f.a_form_cs,
             im.country AS match_country
      FROM match_features m
      JOIN match_form f
        ON f.date=m.date AND f.home_team=m.home_team AND f.away_team=m.away_team
      JOIN intl_matches im
        ON im.date=m.date AND im.home_team=m.home_team AND im.away_team=m.away_team
    """, conn)
    df["date"] = pd.to_datetime(df["date"])
    df["form_gd_diff"] = df["h_form_gd"] - df["a_form_gd"]
    df["form_cs_diff"] = df["h_form_cs"] - df["a_form_cs"]
    df["t_low"] = df["tournament"].str.lower()

    df["h_cont"] = df["home_team"].map(continent_of)
    df["a_cont"] = df["away_team"].map(continent_of)
    df["m_cont"] = df["match_country"].map(continent_of)

    df["h_continent_match"] = (df["h_cont"] == df["m_cont"]).astype(int)
    df["a_continent_match"] = (df["a_cont"] == df["m_cont"]).astype(int)
    df["continent_diff"]    = df["h_continent_match"] - df["a_continent_match"]
    df["is_host_h"] = (df["home_team"] == df["match_country"]).astype(int)
    df["is_host_a"] = (df["away_team"] == df["match_country"]).astype(int)
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
    df = load(conn)

    # Diagnostic: coverage of continent mapping
    unmapped = df[df["h_cont"] == "OTHER"]["home_team"].value_counts().head(15)
    print(f"Total matches: {len(df)}")
    print(f"home_continent_match=1 share: {df['h_continent_match'].mean():.3f}")
    print(f"is_host_h=1 share: {df['is_host_h'].mean():.3f}")
    if len(unmapped) > 0:
        print(f"\nTop unmapped home teams (continent='OTHER'):")
        print(unmapped.to_string())

    df = df.dropna(subset=["form_gd_diff", "form_cs_diff"])

    big = ["fifa world cup", "uefa euro", "copa américa", "copa america",
           "african cup of nations", "afc asian cup"]
    test_pool = df[df["t_low"].isin(big) & (df["date"] >= "2014-01-01")]
    print(f"\nTest pool: {len(test_pool)}")
    years = [2014, 2016, 2018, 2020, 2022, 2024]

    base = ["elo_diff", "home_advantage", "form_gd_diff", "form_cs_diff"]
    variants = {
        "v2.9 (base)":           base,
        "v5a +continent_diff":   base + ["continent_diff"],
        "v5b +is_host":          base + ["is_host_h", "is_host_a"],
        "v5c +cont+host_h":      base + ["continent_diff", "is_host_h"],
        "v5d full (5 new)":      base + ["h_continent_match", "a_continent_match", "is_host_h", "is_host_a"],
        "v5e cont_match h+a":    base + ["h_continent_match", "a_continent_match"],
    }

    print(f"\n{'variant':25s}  " + "  ".join(f"{y:6d}" for y in years) + f"   {'mean':>6s}  {'brier':>7s}")
    print("-" * 100)
    for label, feats in variants.items():
        m_acc, m_b, accs = run(df, test_pool, years, feats)
        accs_str = "  ".join(f"{a:.3f}" if not np.isnan(a) else " --- " for a in accs)
        print(f"{label:25s}  {accs_str}   {m_acc:.3f}   {m_b:.4f}")

    conn.close()


if __name__ == "__main__":
    main()
