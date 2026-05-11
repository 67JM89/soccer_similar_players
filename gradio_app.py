"""
Gradio UI for 2026 World Cup predictions — Friendly Edition.

Improvements vs prior version:
  - Plotly charts (browser renders flag emojis natively)
  - Light, warm color scheme (cream + soft blue + gold)
  - Confederation legend displayed prominently
  - Player card disclaimer about FIFA 23 (2022-23 season) data freshness
  - Better proportions, more whitespace
"""
import sys, io, sqlite3
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr

from wc_predictor import Predictor

DB_PATH = Path(__file__).parent / "data" / "soccer.db"
predictor = Predictor()
WC_TEAMS = predictor.list_teams(wc2026_only=True)


# ============================== Color Palette ==============================
PAL = {
    "bg":         "#fefcf7",      # warm cream
    "card":       "#ffffff",
    "primary":    "#5b8ec4",      # soft sky blue
    "primary_d":  "#3d6c9e",
    "accent":     "#e6b800",      # warm gold
    "accent_l":   "#f4d35e",
    "muted":      "#8b8680",
    "text":       "#2b2b2b",
    "soft_bg":    "#faf6ec",
    "border":     "#ebe5d8",
    "podium_g":   "#d4af37",      # gold
    "podium_s":   "#b8b8b8",      # silver
    "podium_b":   "#cd7f32",      # bronze
}

FLAGS = {
    "Algeria":"🇩🇿","Argentina":"🇦🇷","Australia":"🇦🇺","Austria":"🇦🇹",
    "Belgium":"🇧🇪","Bosnia and Herzegovina":"🇧🇦","Brazil":"🇧🇷",
    "Canada":"🇨🇦","Cape Verde":"🇨🇻","Colombia":"🇨🇴","Croatia":"🇭🇷",
    "Curaçao":"🇨🇼","Czech Republic":"🇨🇿","DR Congo":"🇨🇩",
    "Ecuador":"🇪🇨","Egypt":"🇪🇬","England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "France":"🇫🇷","Germany":"🇩🇪","Ghana":"🇬🇭","Haiti":"🇭🇹",
    "Iran":"🇮🇷","Iraq":"🇮🇶","Ivory Coast":"🇨🇮",
    "Japan":"🇯🇵","Jordan":"🇯🇴","Mexico":"🇲🇽","Morocco":"🇲🇦",
    "Netherlands":"🇳🇱","New Zealand":"🇳🇿","Norway":"🇳🇴",
    "Panama":"🇵🇦","Paraguay":"🇵🇾","Portugal":"🇵🇹",
    "Qatar":"🇶🇦","Saudi Arabia":"🇸🇦","Scotland":"🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "Senegal":"🇸🇳","South Africa":"🇿🇦","South Korea":"🇰🇷",
    "Spain":"🇪🇸","Sweden":"🇸🇪","Switzerland":"🇨🇭",
    "Tunisia":"🇹🇳","Turkey":"🇹🇷","United States":"🇺🇸",
    "Uruguay":"🇺🇾","Uzbekistan":"🇺🇿",
}

CONFED_INFO = {
    "UEFA":     {"color":"#4a72a8", "icon":"🛡️", "name":"UEFA (유럽)"},
    "CONMEBOL": {"color":"#e6b800", "icon":"⭐", "name":"CONMEBOL (남미)"},
    "CONCACAF": {"color":"#6abf69", "icon":"🌎", "name":"CONCACAF (북·중미·카리브)"},
    "AFC":      {"color":"#c44d4d", "icon":"🏯", "name":"AFC (아시아·호주)"},
    "CAF":      {"color":"#8c564b", "icon":"🦁", "name":"CAF (아프리카)"},
    "OFC":      {"color":"#7a4eb3", "icon":"🌊", "name":"OFC (오세아니아)"},
}
CONFED_OF = {
    **{t:"UEFA" for t in ["Spain","France","England","Germany","Netherlands","Portugal","Belgium",
                          "Croatia","Switzerland","Austria","Norway","Turkey","Czech Republic",
                          "Bosnia and Herzegovina","Sweden","Scotland"]},
    **{t:"CONMEBOL" for t in ["Argentina","Brazil","Colombia","Ecuador","Uruguay","Paraguay"]},
    **{t:"CONCACAF" for t in ["United States","Mexico","Canada","Panama","Haiti","Curaçao"]},
    **{t:"AFC" for t in ["Japan","South Korea","Iran","Iraq","Saudi Arabia","Qatar","Uzbekistan",
                         "Australia","Jordan"]},
    **{t:"CAF" for t in ["Morocco","Senegal","Egypt","Ivory Coast","Ghana","Algeria","DR Congo",
                         "Tunisia","South Africa","Cape Verde"]},
    **{t:"OFC" for t in ["New Zealand"]},
}


def flag(t): return FLAGS.get(t, "🏳️")
def conf_of(t): return CONFED_OF.get(t, "—")
def conf_color(t): return CONFED_INFO.get(conf_of(t), {}).get("color", "#888")


def rating_color(v):
    if v >= 85: return "#1b5e20"
    if v >= 75: return "#388e3c"
    if v >= 65: return "#7cb342"
    if v >= 55: return "#fbc02d"
    if v >= 45: return "#f57c00"
    return "#d32f2f"


# Plotly base styling
PLOTLY_FONT = dict(family="'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', system-ui, sans-serif",
                   color=PAL["text"], size=13)
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=PLOTLY_FONT,
    margin=dict(l=10, r=10, t=40, b=10),
)


# ============================== Charts (Plotly) ==============================
def fig_score_heatmap(grid, team_a, team_b):
    g = (grid[:6, :6] * 100).astype(float)
    text = [[f"{v:.1f}" for v in row] for row in g]
    fig = go.Figure(data=go.Heatmap(
        z=g, text=text, texttemplate="%{text}", textfont={"size": 11},
        colorscale="Viridis", showscale=True, colorbar=dict(title="P %", thickness=12),
        x=[f"{i}" for i in range(6)], y=[f"{i}" for i in range(6)],
        hovertemplate=f"{flag(team_a)} {team_a} %{{y}}-%{{x}} {flag(team_b)} {team_b}<br>P = %{{z:.2f}}%<extra></extra>",
    ))
    fig.update_layout(
        title=f"{flag(team_a)} {team_a} goals  vs  {flag(team_b)} {team_b} goals",
        xaxis_title=f"{flag(team_b)} {team_b} goals",
        yaxis_title=f"{flag(team_a)} {team_a} goals",
        yaxis=dict(autorange="reversed"),
        height=420, **PLOTLY_LAYOUT,
    )
    return fig


def fig_winprob_donut(p_a, p_d, p_b, team_a, team_b):
    labels = [f"{flag(team_a)} {team_a}<br>{p_a*100:.1f}%",
              f"Draw<br>{p_d*100:.1f}%",
              f"{flag(team_b)} {team_b}<br>{p_b*100:.1f}%"]
    colors = [conf_color(team_a), "#cccccc", conf_color(team_b)]
    fav = team_a if p_a > p_b else team_b
    fav_p = max(p_a, p_b)
    fig = go.Figure(data=go.Pie(
        values=[p_a, p_d, p_b], labels=labels, hole=0.55,
        marker=dict(colors=colors, line=dict(color="white", width=4)),
        textfont=dict(size=14), textposition="outside", showlegend=False,
        hovertemplate="%{label}<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{flag(fav)}</b><br><span style='font-size:14px'>{fav_p*100:.1f}%</span>",
        x=0.5, y=0.5, font=dict(size=32), showarrow=False,
    )
    fig.add_annotation(
        text="FAVORED", x=0.5, y=0.62, font=dict(size=10, color=PAL["muted"]),
        showarrow=False,
    )
    fig.update_layout(
        title="<b>Win / Draw / Win</b>", title_x=0.5, height=400,
        **PLOTLY_LAYOUT
    )
    return fig


def fig_top_scores_bar(top_scores, team_a, team_b):
    scores = [s["score"] for s in top_scores]
    probs = [s["prob"] * 100 for s in top_scores]
    fig = go.Figure(go.Bar(
        x=probs, y=scores, orientation="h",
        marker=dict(color=probs, colorscale="Viridis", line=dict(width=0)),
        text=[f"{p:.2f}%" for p in probs], textposition="outside",
        hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title=f"<b>Top 5 most likely scorelines</b><br><sub>{flag(team_a)} {team_a} vs {flag(team_b)} {team_b}</sub>",
        xaxis_title="Probability %",
        yaxis=dict(autorange="reversed", tickfont=dict(size=15, family="monospace")),
        height=380, showlegend=False, **PLOTLY_LAYOUT,
    )
    return fig


def fig_outright_bar(top_n=20):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        'SELECT team, "P(Champion)" AS p FROM wc2026_predictions_v3 '
        'ORDER BY p DESC LIMIT ?', conn, params=(top_n,))
    conn.close()
    df["p_pct"] = df["p"] * 100
    df["confed"] = df["team"].map(conf_of)
    df["color"] = df["team"].map(conf_color)
    df["label"] = df.apply(lambda r: f"{flag(r['team'])} {r['team']}", axis=1)

    fig = go.Figure()
    for confed, info in CONFED_INFO.items():
        sub = df[df["confed"] == confed]
        if len(sub) == 0: continue
        fig.add_trace(go.Bar(
            x=sub["p_pct"], y=sub["label"], orientation="h",
            marker=dict(color=info["color"], line=dict(width=0)),
            name=f"{info['icon']} {confed}",
            text=sub["p_pct"].apply(lambda v: f"{v:.2f}%"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>P(Champion) = %{x:.2f}%<extra></extra>",
        ))
    fig.update_layout(
        title="<b>Championship Probability</b>",
        xaxis_title="Probability %",
        yaxis=dict(categoryorder="array", categoryarray=df["label"][::-1].tolist(),
                   tickfont=dict(size=12)),
        barmode="overlay",
        height=max(420, top_n * 28),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.05, x=0.5, xanchor="center",
                    bgcolor="rgba(255,255,255,0.6)", bordercolor=PAL["border"], borderwidth=1,
                    font=dict(size=11)),
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("margin",)},
        margin=dict(l=10, r=60, t=50, b=80),
    )
    return fig


def fig_stage_heatmap():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT team, "P(advance)" AS adv, "P(R16)" AS r16,
               "P(QF)" AS qf, "P(SF)" AS sf, "P(Final)" AS fin, "P(Champion)" AS win
        FROM wc2026_predictions_v3 ORDER BY win DESC
    """, conn)
    conn.close()
    M = (df[["adv","r16","qf","sf","fin","win"]].values * 100).astype(float)
    labels = [f"{flag(t)} {t}" for t in df["team"]]
    stages = ["Group↑", "R16", "QF", "SF", "Final", "🏆 Win"]
    text = [[(f"{v:.0f}" if v >= 10 else f"{v:.1f}") if v >= 1 else "" for v in row] for row in M]

    fig = go.Figure(data=go.Heatmap(
        z=M, x=stages, y=labels, text=text, texttemplate="%{text}",
        textfont={"size": 11},
        colorscale=[[0, "#fff8e1"], [0.5, "#ffb300"], [1, "#bf360c"]],
        zmin=0, zmax=100, showscale=True,
        colorbar=dict(title="P %", thickness=12),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="<b>Stage Progression Probabilities</b>",
        height=900, yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        **PLOTLY_LAYOUT,
    )
    return fig


def fig_group_visualizer():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT g."group" AS grp, g.team, v3."P(advance)" AS p_adv
        FROM wc2026_groups g
        LEFT JOIN wc2026_predictions_v3 v3 ON v3.team = g.team
        ORDER BY grp, g.draw_position
    """, conn)
    conn.close()
    df["p_adv"] = df["p_adv"] * 100
    groups = sorted(df["grp"].unique())

    fig = make_subplots(rows=4, cols=3, subplot_titles=[f"Group {g}" for g in groups],
                        horizontal_spacing=0.12, vertical_spacing=0.08)
    for idx, grp in enumerate(groups):
        sub = df[df["grp"] == grp].sort_values("p_adv", ascending=True).reset_index(drop=True)
        row, col = idx // 3 + 1, idx % 3 + 1
        labels = [f"{flag(t)} {t}" for t in sub["team"]]
        colors = [conf_color(t) for t in sub["team"]]
        fig.add_trace(go.Bar(
            x=sub["p_adv"], y=labels, orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=sub["p_adv"].apply(lambda v: f"{v:.0f}%"),
            textposition="outside",
            hovertemplate="<b>%{y}</b>: %{x:.1f}%<extra></extra>",
            showlegend=False,
        ), row=row, col=col)
        fig.update_xaxes(range=[0, 120], row=row, col=col, showticklabels=False)
        fig.update_yaxes(tickfont=dict(size=10), row=row, col=col)

    fig.update_layout(
        height=1000,
        title="<b>Group Stage  ·  P(advance to knockout) — color = confederation</b>",
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"},
        margin=dict(l=10, r=20, t=70, b=20),
    )
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=13, color=PAL["primary_d"])
    return fig


# ============================== Confederation Legend ==============================
def render_confed_legend(title: str = "🌐 컨페더레이션 색상 가이드") -> str:
    items = []
    for c, info in CONFED_INFO.items():
        items.append(f"""
        <div style="display:flex; align-items:center; gap:6px; padding:6px 10px; background:white;
                    border-radius:8px; border:1px solid {PAL['border']};">
            <span style="width:14px; height:14px; border-radius:4px; background:{info['color']};
                        flex-shrink:0;"></span>
            <span style="font-size:12px; color:{PAL['text']};">
                <b>{info['icon']} {c}</b> · {info['name'].split('(')[1].rstrip(')')}
            </span>
        </div>
        """)
    return f"""
    <div style="background:{PAL['soft_bg']}; padding:10px 14px; border-radius:10px;
                margin:8px 0 14px; border:1px solid {PAL['border']};">
        <div style="font-size:11px; letter-spacing:1.5px; color:{PAL['muted']}; margin-bottom:6px;
                    font-weight:600;">{title}</div>
        <div style="display:flex; gap:8px; flex-wrap:wrap;">
            {''.join(items)}
        </div>
    </div>
    """


# ============================== Top 3 Podium ==============================
def render_podium() -> str:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        'SELECT team, "P(Champion)" AS p FROM wc2026_predictions_v3 '
        'ORDER BY p DESC LIMIT 3', conn)
    conn.close()
    t1, t2, t3 = df.iloc[0], df.iloc[1], df.iloc[2]

    def step(row, medal_color, label, height_px):
        team = row["team"]; p = row["p"] * 100
        return f"""
        <div style="display:flex; flex-direction:column; align-items:center; flex:1; padding:6px;">
            <div style="font-size:58px; line-height:1;">{flag(team)}</div>
            <div style="font-size:16px; font-weight:700; margin:10px 0 0; color:{PAL['text']}; text-align:center;">
                {team}
            </div>
            <div style="font-size:24px; font-weight:900; color:{medal_color}; margin:6px 0 10px;">
                {p:.1f}%
            </div>
            <div style="width:100%; height:{height_px}px;
                        background:linear-gradient(180deg,{medal_color},{medal_color}cc);
                        border-radius:10px 10px 0 0; display:flex; align-items:center; justify-content:center;
                        color:white; font-size:34px; font-weight:900;
                        text-shadow:0 2px 4px rgba(0,0,0,0.2);">{label}</div>
        </div>
        """

    return f"""
    <div style="background:linear-gradient(135deg, #fffdf6, #faf3dd);
                padding:18px 24px; border-radius:14px; margin-bottom:14px;
                border:1px solid {PAL['border']};">
        <div style="text-align:center; font-size:11px; color:{PAL['muted']};
                    letter-spacing:2.5px; font-weight:600; margin-bottom:8px;">
            🏆 PREDICTED PODIUM
        </div>
        <div style="text-align:center; font-size:11px; color:{PAL['muted']}; margin-bottom:14px;">
            Monte Carlo N=10,000 시뮬레이션 결과
        </div>
        <div style="display:flex; align-items:flex-end; gap:6px; max-width:620px; margin:0 auto;">
            {step(t2, PAL['podium_s'], "2", 100)}
            {step(t1, PAL['podium_g'], "1", 140)}
            {step(t3, PAL['podium_b'], "3", 70)}
        </div>
    </div>
    """


# ============================== Hero ==============================
def render_hero() -> str:
    conn = sqlite3.connect(DB_PATH)
    n_players = pd.read_sql("SELECT COUNT(*) c FROM players", conn).c[0]
    n_matches = pd.read_sql("SELECT COUNT(*) c FROM intl_matches", conn).c[0]
    conn.close()
    return f"""
    <div style="background:linear-gradient(135deg,#fffdf6 0%,#faf3dd 100%);
                color:{PAL['text']}; padding:24px 28px; border-radius:16px; margin-bottom:14px;
                border:1px solid {PAL['border']};">
        <div style="display:flex; align-items:center; gap:14px; flex-wrap:wrap;">
            <div style="font-size:44px; line-height:1;">🏆</div>
            <div>
                <div style="font-size:26px; font-weight:800; letter-spacing:-0.3px; color:{PAL['text']};">
                    2026 FIFA World Cup Predictor
                </div>
                <div style="opacity:0.7; margin-top:4px; font-size:13px;">
                    Probabilistic predictions for every team and matchup ·
                    <b style="color:{PAL['primary_d']};">v3.5</b> model
                </div>
            </div>
        </div>
        <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
                    gap:10px; margin-top:18px;">
            <div style="background:white; padding:12px 16px; border-radius:12px;
                        border:1px solid {PAL['border']};">
                <div style="font-size:10px; letter-spacing:1.4px; color:{PAL['muted']}; font-weight:600;">QUALIFIED TEAMS</div>
                <div style="font-size:26px; font-weight:800; margin-top:2px; color:{PAL['primary_d']};">48</div>
            </div>
            <div style="background:white; padding:12px 16px; border-radius:12px;
                        border:1px solid {PAL['border']};">
                <div style="font-size:10px; letter-spacing:1.4px; color:{PAL['muted']}; font-weight:600;">TRAINING MATCHES</div>
                <div style="font-size:26px; font-weight:800; margin-top:2px; color:{PAL['primary_d']};">{n_matches:,}</div>
            </div>
            <div style="background:white; padding:12px 16px; border-radius:12px;
                        border:1px solid {PAL['border']};">
                <div style="font-size:10px; letter-spacing:1.4px; color:{PAL['muted']}; font-weight:600;">PLAYER RATINGS</div>
                <div style="font-size:26px; font-weight:800; margin-top:2px; color:{PAL['primary_d']};">{n_players:,}</div>
            </div>
            <div style="background:white; padding:12px 16px; border-radius:12px;
                        border:1px solid {PAL['border']};">
                <div style="font-size:10px; letter-spacing:1.4px; color:{PAL['muted']}; font-weight:600;">BACKTEST ACCURACY</div>
                <div style="font-size:26px; font-weight:800; margin-top:2px; color:{PAL['accent']};">~55%</div>
            </div>
        </div>
    </div>
    """


# ============================== Player Cards ==============================
def get_team_squad(team: str, top_n: int = 25, position_filter: str = "All") -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    al = pd.read_sql("SELECT canonical, fifa_name FROM nation_aliases", conn)
    alias = dict(zip(al["canonical"], al["fifa_name"]))
    fifa_name = alias.get(team, team)
    df = pd.read_sql("""
        SELECT short_name, long_name, club_name, league_name, age,
               height_cm, weight_kg, player_positions, preferred_foot,
               overall, potential, value_eur, wage_eur, is_gk,
               atk_finishing, atk_shooting, atk_dribbling, atk_vision,
               def_tackling, def_marking, def_interceptions, def_aggression,
               phy_pace, phy_strength, phy_stamina, phy_jumping,
               gkdist_kicking, gkdist_handling, gkdist_positioning, gkdist_reflexes,
               gksave_diving, gksave_handling, gksave_reflexes, gksave_speed,
               score_atk, score_def, score_phy, score_gk_dist, score_gk_save
        FROM players
        WHERE LOWER(nationality_name) = LOWER(?)
        ORDER BY overall DESC LIMIT 40
    """, conn, params=(fifa_name,))
    conn.close()
    if position_filter == "GK":
        df = df[df["is_gk"] == 1]
    elif position_filter in ("DEF", "MID", "ATT"):
        groups = {"DEF":["CB","LB","RB","LWB","RWB"], "MID":["CM","CDM","CAM","LM","RM"],
                  "ATT":["ST","CF","LW","RW"]}
        df = df[df["is_gk"] == 0]
        df = df[df["player_positions"].fillna("").apply(
            lambda p: any(x in p for x in groups[position_filter]))]
    return df.head(top_n).reset_index(drop=True)


def stat_bar(name, value):
    v = int(round(float(value))) if not pd.isna(value) else 0
    pct = max(0, min(100, v))
    c = rating_color(v)
    return f"""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px; font-size:12px;">
        <div style="flex:0 0 95px; color:#555;">{name}</div>
        <div style="flex:0 0 28px; text-align:right; color:{c}; font-weight:700;">{v}</div>
        <div style="flex:1; background:#eee; border-radius:4px; height:7px; overflow:hidden;">
            <div style="background:{c}; width:{pct}%; height:100%; border-radius:4px;"></div>
        </div>
    </div>
    """


def render_card(row):
    is_gk = int(row["is_gk"]) == 1
    name = row["short_name"] or row["long_name"]
    overall = int(row["overall"])
    oc = rating_color(overall)
    club = row["club_name"] or "—"
    age = int(row["age"]) if not pd.isna(row["age"]) else "—"
    position = (row["player_positions"] or "—").split(",")[0].strip()
    foot = row["preferred_foot"] or ""
    height = f"{int(row['height_cm'])}cm" if not pd.isna(row["height_cm"]) else ""

    if is_gk:
        c1 = ("🥅 Distribution", row["score_gk_dist"], [
            ("Kicking", row["gkdist_kicking"]), ("Handling", row["gkdist_handling"]),
            ("Positioning", row["gkdist_positioning"]), ("Reflexes", row["gkdist_reflexes"])])
        c2 = ("✋ Saves", row["score_gk_save"], [
            ("Diving", row["gksave_diving"]), ("Handling", row["gksave_handling"]),
            ("Reflexes", row["gksave_reflexes"]), ("Speed", row["gksave_speed"])])
    else:
        c1 = ("⚔️ Attack", row["score_atk"], [
            ("Finishing", row["atk_finishing"]), ("Shooting", row["atk_shooting"]),
            ("Dribbling", row["atk_dribbling"]), ("Vision", row["atk_vision"])])
        c2 = ("🛡️ Defense", row["score_def"], [
            ("Tackling", row["def_tackling"]), ("Marking", row["def_marking"]),
            ("Interceptions", row["def_interceptions"]), ("Aggression", row["def_aggression"])])
    c3 = ("💪 Physical", row["score_phy"], [
        ("Pace", row["phy_pace"]), ("Strength", row["phy_strength"]),
        ("Stamina", row["phy_stamina"]), ("Jumping", row["phy_jumping"])])

    def sect(name, avg, stats):
        av = int(round(float(avg))) if not pd.isna(avg) else 0
        ac = rating_color(av)
        bars = "".join(stat_bar(n, v) for n, v in stats)
        return f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between;
                        border-bottom:1px solid #eee; margin-bottom:6px; padding-bottom:3px;">
                <span style="font-weight:700; font-size:12px;">{name}</span>
                <span style="color:{ac}; font-weight:700; font-size:13px;">{av}</span>
            </div>{bars}
        </div>
        """

    return f"""
    <div style="background:white; border-radius:12px; padding:14px;
                box-shadow:0 2px 8px rgba(91,142,196,0.10); border-top:5px solid {oc};">
        <div style="display:flex; gap:12px; margin-bottom:12px;">
            <div style="background:{oc}; color:white; width:56px; height:56px;
                        border-radius:10px; display:flex; align-items:center; justify-content:center;
                        font-size:24px; font-weight:900; flex-shrink:0;
                        box-shadow:0 2px 4px rgba(0,0,0,0.12);">{overall}</div>
            <div style="flex:1; min-width:0;">
                <div style="font-size:14px; font-weight:700; color:#222;
                            overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{name}</div>
                <div style="font-size:11px; color:#666; margin-top:2px;
                            overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{club}</div>
                <div style="font-size:10px; color:#888; margin-top:3px;">
                    <b>{position}</b> · {age}y · {height}{(' · ' + foot[0]) if foot else ''}
                </div>
            </div>
        </div>
        {sect(*c1)}{sect(*c2)}{sect(*c3)}
    </div>
    """


def render_team_squad(team, position_filter):
    df = get_team_squad(team, top_n=25, position_filter=position_filter)
    if len(df) == 0:
        return f"<p style='padding:20px;'>No players for {team} ({position_filter})</p>"
    cards = "".join(render_card(row) for _, row in df.iterrows())
    n_gk = int(df["is_gk"].sum())
    avg = df["overall"].mean()
    confed = conf_of(team)
    cinfo = CONFED_INFO.get(confed, {"color":"#888","icon":"","name":""})

    header = f"""
    <div style="background:linear-gradient(135deg,{cinfo['color']},{cinfo['color']}dd);
                color:white; padding:20px 24px; border-radius:14px; margin-bottom:14px;
                display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
        <div>
            <div style="font-size:32px; font-weight:800;">{flag(team)} {team}</div>
            <div style="font-size:12px; opacity:0.92; margin-top:6px;">
                {cinfo['icon']} {confed} · {cinfo['name'].split('(')[1].rstrip(')') if '(' in cinfo['name'] else ''}
            </div>
        </div>
        <div style="display:flex; gap:18px;">
            <div><div style="font-size:10px; opacity:0.8; letter-spacing:1px;">PLAYERS</div>
                <div style="font-size:24px; font-weight:800;">{len(df)}</div></div>
            <div><div style="font-size:10px; opacity:0.8; letter-spacing:1px;">AVG OVR</div>
                <div style="font-size:24px; font-weight:800;">{avg:.1f}</div></div>
            <div><div style="font-size:10px; opacity:0.8; letter-spacing:1px;">GK</div>
                <div style="font-size:24px; font-weight:800;">{n_gk}</div></div>
        </div>
    </div>
    <div style="background:#fff8e1; border-left:4px solid {PAL['accent']};
                padding:10px 14px; border-radius:8px; margin-bottom:14px; font-size:12px; color:#5a4a00;">
        📅 <b>Data source: FIFA 23 (2022-23 시즌)</b> — 일부 선수의 소속 클럽은 최신과 다를 수 있어요.
        국가대표 자격은 거의 안 바뀌므로 WC 예측엔 유효.
    </div>
    """
    grid = f"""<div style="display:grid; grid-template-columns:repeat(auto-fill,minmax(265px,1fr)); gap:12px;">{cards}</div>"""
    return header + grid


# ============================== Bracket ==============================
def simulate_one_bracket(seed: int = 42):
    rng = np.random.default_rng(int(seed))
    conn = sqlite3.connect(DB_PATH)
    groups_df = pd.read_sql('SELECT team, "group" AS grp FROM wc2026_groups', conn)
    conn.close()
    groups = {g: list(groups_df[groups_df["grp"] == g]["team"]) for g in sorted(groups_df["grp"].unique())}

    def sim(ta, tb, ko=False):
        lh, la = predictor.expected_goals(ta, tb, neutral=True)
        sa = int(rng.poisson(lh)); sb = int(rng.poisson(la))
        w = ta if sa > sb else tb if sb > sa else None
        if ko and w is None:
            sa += int(rng.poisson(lh / 3)); sb += int(rng.poisson(la / 3))
            w = ta if sa > sb else tb if sb > sa else None
            if w is None: w = ta if rng.random() < 0.5 else tb
        return ta, tb, sa, sb, w

    grp_res = {}
    firsts, seconds, thirds = [], [], []
    for g, teams in groups.items():
        st = {t: {"team": t, "pts": 0, "gf": 0, "ga": 0, "gd": 0} for t in teams}
        ms = []
        for i in range(len(teams)):
            for j in range(i+1, len(teams)):
                ta, tb, sa, sb, _ = sim(teams[i], teams[j])
                ms.append((ta, tb, sa, sb))
                st[ta]["gf"]+=sa; st[ta]["ga"]+=sb
                st[tb]["gf"]+=sb; st[tb]["ga"]+=sa
                if sa>sb: st[ta]["pts"]+=3
                elif sa<sb: st[tb]["pts"]+=3
                else: st[ta]["pts"]+=1; st[tb]["pts"]+=1
        for v in st.values(): v["gd"] = v["gf"] - v["ga"]
        ranked = sorted(st.values(), key=lambda v: (-v["pts"], -v["gd"], -v["gf"], rng.random()))
        for r in ranked: r["group"] = g
        firsts.append(ranked[0]); seconds.append(ranked[1]); thirds.append(ranked[2])
        grp_res[g] = {"ranked": ranked, "matches": ms}

    thirds_sorted = sorted(thirds, key=lambda v: (-v["pts"], -v["gd"], -v["gf"], rng.random()))[:8]
    advancing = [s["team"] for s in firsts + seconds + thirds_sorted]
    rng.shuffle(advancing)

    rounds = {}
    cur = advancing
    for label in ["R32", "R16", "QF", "SF"]:
        nxt, res = [], []
        for i in range(0, len(cur), 2):
            ta, tb, sa, sb, w = sim(cur[i], cur[i+1], ko=True)
            res.append((ta, tb, sa, sb, w)); nxt.append(w)
        rounds[label] = res; cur = nxt
    ta, tb, sa, sb, w = sim(cur[0], cur[1], ko=True)
    rounds["Final"] = [(ta, tb, sa, sb, w)]
    return {"groups": grp_res, "rounds": rounds,
            "champion": w, "runner": tb if ta == w else ta}


def render_match_block(home, away, sh, sa, winner):
    h_win = winner == home; a_win = winner == away
    h_bg = "#e6f4ea" if h_win else "white"
    a_bg = "#e6f4ea" if a_win else "white"
    return f"""
    <div style="background:white; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.06);
                overflow:hidden; min-width:165px; margin:3px 0;">
        <div style="display:flex; align-items:center; padding:6px 9px; background:{h_bg}; border-bottom:1px solid #f5f5f5;">
            <span style="font-size:15px; margin-right:6px;">{flag(home)}</span>
            <span style="flex:1; font-size:12px; font-weight:{'700' if h_win else '500'};
                        color:{'#1b5e20' if h_win else '#444'}; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{home}</span>
            <span style="font-weight:700; font-size:14px; color:{'#1b5e20' if h_win else '#444'};">{sh}</span>
        </div>
        <div style="display:flex; align-items:center; padding:6px 9px; background:{a_bg};">
            <span style="font-size:15px; margin-right:6px;">{flag(away)}</span>
            <span style="flex:1; font-size:12px; font-weight:{'700' if a_win else '500'};
                        color:{'#1b5e20' if a_win else '#444'}; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{away}</span>
            <span style="font-weight:700; font-size:14px; color:{'#1b5e20' if a_win else '#444'};">{sa}</span>
        </div>
    </div>
    """


def render_bracket(result):
    champion = result["champion"]; runner = result["runner"]
    rounds_data = [("Round of 32", result["rounds"]["R32"]),
                   ("Round of 16", result["rounds"]["R16"]),
                   ("Quarter-Finals", result["rounds"]["QF"]),
                   ("Semi-Finals", result["rounds"]["SF"]),
                   ("Final", result["rounds"]["Final"])]
    cols = []
    for title, matches in rounds_data:
        blocks = "".join(render_match_block(h,a,sh,sa,w) for h,a,sh,sa,w in matches)
        cols.append(f"""
        <div style="display:flex; flex-direction:column; min-width:180px;">
            <div style="font-size:11px; font-weight:700; color:{PAL['primary_d']};
                        letter-spacing:1.4px; text-align:center; padding:6px;
                        border-bottom:2px solid {PAL['primary']}; margin-bottom:8px;">{title}</div>
            <div style="display:flex; flex-direction:column; justify-content:space-around; flex:1; gap:4px;">
                {blocks}
            </div>
        </div>
        """)

    champ_card = f"""
    <div style="background:linear-gradient(135deg,#fff3a6,{PAL['accent']});
                border-radius:16px; padding:26px 20px; text-align:center; color:#4a3500;
                box-shadow:0 8px 28px rgba(230,184,0,0.32); margin-bottom:18px;">
        <div style="font-size:11px; letter-spacing:3px; font-weight:700; margin-bottom:8px;">🏆 CHAMPION</div>
        <div style="font-size:64px; line-height:1;">{flag(champion)}</div>
        <div style="font-size:28px; font-weight:900; margin:10px 0 4px;">{champion}</div>
        <div style="font-size:12px; opacity:0.8;">Defeated {flag(runner)} {runner}</div>
    </div>
    """
    return f"""
    {champ_card}
    <div style="background:linear-gradient(180deg,{PAL['soft_bg']},#ffffff);
                padding:18px 12px; border-radius:14px; border:1px solid {PAL['border']};">
        <div style="display:flex; gap:14px; overflow-x:auto; align-items:stretch;">
            {''.join(cols)}
        </div>
    </div>
    """


def render_groups(result):
    html = []
    for g in sorted(result["groups"].keys()):
        ranked = result["groups"][g]["ranked"]
        rows = []
        for pos, t in enumerate(ranked, 1):
            badge, row_bg = "", ""
            if pos == 1:
                badge = '<span style="background:#2e7d32; color:white; font-size:9px; padding:1px 6px; border-radius:8px;">1st</span>'
                row_bg = "background:#e8f5e9;"
            elif pos == 2:
                badge = '<span style="background:#558b2f; color:white; font-size:9px; padding:1px 6px; border-radius:8px;">2nd</span>'
                row_bg = "background:#f1f8e9;"
            elif pos == 3:
                badge = '<span style="background:#f9a825; color:#4a3500; font-size:9px; padding:1px 6px; border-radius:8px;">3rd</span>'
                row_bg = "background:#fff8e1;"
            rows.append(f"""
            <tr style="{row_bg}">
                <td style="padding:3px 6px; text-align:center; color:#888; font-size:11px;">{pos}</td>
                <td style="padding:3px 6px; font-size:12px;"><span style="margin-right:4px;">{flag(t['team'])}</span>{t['team']}</td>
                <td style="padding:3px 6px; text-align:center;">{badge}</td>
                <td style="padding:3px 6px; text-align:center; font-weight:700; font-size:12px;">{t['pts']}</td>
                <td style="padding:3px 6px; text-align:center; font-size:11px; color:#666;">{t['gd']:+d}</td>
            </tr>""")
        html.append(f"""
        <div style="background:white; padding:10px 12px; border-radius:10px;
                    border:1px solid {PAL['border']};
                    border-top:3px solid {conf_color(ranked[0]['team'])};">
            <div style="font-weight:700; color:{PAL['primary_d']}; margin-bottom:6px; font-size:13px;">Group {g}</div>
            <table style="width:100%; border-collapse:collapse;">
                <tr style="color:#aaa; font-size:10px;">
                    <td></td><td>Team</td><td></td><td style="text-align:center;">Pts</td><td style="text-align:center;">GD</td>
                </tr>
                {''.join(rows)}
            </table>
        </div>""")
    return f"""<div style="display:grid; grid-template-columns:repeat(auto-fill,minmax(245px,1fr)); gap:12px; margin-top:10px;">{''.join(html)}</div>"""


def run_bracket(seed):
    r = simulate_one_bracket(int(seed))
    return render_bracket(r), render_groups(r)


# ============================== Match Predictor ==============================
def render_match_summary(team_a, team_b, r):
    confed_a = conf_of(team_a); confed_b = conf_of(team_b)
    info_a = CONFED_INFO.get(confed_a, {"color":"#888","icon":""})
    info_b = CONFED_INFO.get(confed_b, {"color":"#888","icon":""})
    elo_diff = r["elo_a"] - r["elo_b"]

    def side(team, info, elo):
        return f"""
        <div style="flex:1; text-align:center; padding:14px;">
            <div style="font-size:54px; line-height:1;">{flag(team)}</div>
            <div style="font-size:18px; font-weight:800; margin-top:8px; color:{PAL['text']};">{team}</div>
            <div style="display:inline-block; margin-top:8px; padding:3px 10px; border-radius:12px;
                        background:{info['color']}; color:white; font-size:10px; font-weight:600;
                        letter-spacing:0.5px;">{info['icon']} {confed_a if team==team_a else confed_b}</div>
            <div style="font-size:10px; color:{PAL['muted']}; margin-top:10px; letter-spacing:1px;">ELO</div>
            <div style="font-size:22px; font-weight:800; color:{PAL['primary_d']};">{elo:.0f}</div>
        </div>
        """

    return f"""
    <div style="background:white; border-radius:16px; padding:18px 24px; margin-bottom:14px;
                border:1px solid {PAL['border']}; box-shadow:0 2px 10px rgba(91,142,196,0.08);">
        <div style="display:flex; align-items:center; gap:8px;">
            {side(team_a, info_a, r['elo_a'])}
            <div style="text-align:center; padding:0 10px; min-width:160px;">
                <div style="font-size:10px; opacity:0.5; letter-spacing:2px; font-weight:600;">VS</div>
                <div style="font-size:11px; color:{PAL['muted']}; margin-top:8px; letter-spacing:1px;">EXPECTED GOALS</div>
                <div style="font-size:32px; font-weight:900; color:{PAL['primary_d']}; line-height:1.1; margin-top:4px;">
                    {r['expected_a_goals']} : {r['expected_b_goals']}
                </div>
                <div style="font-size:11px; color:{PAL['muted']}; margin-top:8px;">
                    most likely <b style="color:{PAL['accent']};">{r['most_likely_score']}</b>
                </div>
                <div style="font-size:11px; color:{PAL['muted']}; margin-top:4px;">
                    ELO diff <b>{elo_diff:+.0f}</b>
                </div>
            </div>
            {side(team_b, info_b, r['elo_b'])}
        </div>
    </div>
    """


def render_why_panel(team_a, team_b):
    ea = predictor.adjusted_elo(team_a); eb = predictor.adjusted_elo(team_b)
    base_a = predictor._lookup(team_a, predictor.elo, 1500.0)
    base_b = predictor._lookup(team_b, predictor.elo, 1500.0)
    club_a = predictor.club_adjust.get(team_a, 0.0)
    club_b = predictor.club_adjust.get(team_b, 0.0)
    form_gd_a = predictor._lookup(team_a, predictor.form_gd, 0.0)
    form_gd_b = predictor._lookup(team_b, predictor.form_gd, 0.0)
    form_cs_a = predictor._lookup(team_a, predictor.form_cs, 0.0)
    form_cs_b = predictor._lookup(team_b, predictor.form_cs, 0.0)

    def row(label, va, vb, fmt="{:+.0f}", advantage="higher"):
        if advantage == "higher":
            winner = team_a if va > vb else team_b if vb > va else None
        else:
            winner = team_b if va > vb else team_a if vb > va else None
        wc = conf_color(winner) if winner else "#999"
        return f"""
        <tr>
            <td style="padding:8px 12px; font-size:12px; color:#555;">{label}</td>
            <td style="padding:8px 12px; text-align:right; font-size:13px; font-family:monospace;">{fmt.format(va).lstrip('+')}</td>
            <td style="padding:8px 4px; text-align:center; font-size:11px; color:#bbb;">vs</td>
            <td style="padding:8px 12px; font-size:13px; font-family:monospace;">{fmt.format(vb).lstrip('+')}</td>
            <td style="padding:8px 12px; text-align:right; font-size:11px;">
                {f"우위: <b style='color:{wc}'>{flag(winner)} {winner}</b>" if winner else "—"}
            </td>
        </tr>
        """

    return f"""
    <div style="background:white; padding:16px 20px; border-radius:14px;
                border:1px solid {PAL['border']}; box-shadow:0 2px 8px rgba(91,142,196,0.06);">
        <div style="font-size:13px; font-weight:700; color:{PAL['primary_d']};
                    letter-spacing:1px; margin-bottom:10px;">📊 어떤 요소가 이 예측을 만들었나</div>
        <table style="width:100%; border-collapse:collapse;">
            <tr style="border-bottom:1px solid #eee; color:#999; font-size:10px;">
                <td style="padding:4px 12px;">Factor</td>
                <td style="padding:4px 12px; text-align:right;">{flag(team_a)} {team_a}</td>
                <td></td>
                <td style="padding:4px 12px;">{flag(team_b)} {team_b}</td>
                <td></td>
            </tr>
            {row("Base ELO (25년 누적)", base_a, base_b, "{:.0f}")}
            {row("Club ELO 보정 (선수 클럽 강도)", club_a, club_b, "{:+.0f}")}
            {row("최근 폼 (GD/경기, 10경기)", form_gd_a, form_gd_b, "{:+.2f}")}
            {row("최근 폼 (무실점 %)", form_cs_a*100, form_cs_b*100, "{:.0f}%")}
        </table>
    </div>
    """


def predict_match(team_a, team_b, neutral):
    if team_a == team_b:
        return "<div style='padding:20px; color:#c00; text-align:center;'>⚠️ 같은 팀끼리 선택할 수 없어요.</div>", None, None, None, ""
    r = predictor.predict_match(team_a, team_b, neutral=neutral)
    summary = render_match_summary(team_a, team_b, r)
    fig_donut = fig_winprob_donut(r["p_a_win"], r["p_draw"], r["p_b_win"], team_a, team_b)
    grid = np.asarray(r["score_grid"])
    fig_heat = fig_score_heatmap(grid, team_a, team_b)
    fig_top = fig_top_scores_bar(r["top_scores"], team_a, team_b)
    why = render_why_panel(team_a, team_b)
    return summary, fig_donut, fig_heat, fig_top, why


QUICK_PAIRS = {
    "🇪🇸 vs 🇦🇷  (Top 2)": ("Spain", "Argentina"),
    "🇫🇷 vs 🇧🇷  (Big match)": ("France", "Brazil"),
    "🏴 vs 🇩🇪  (Classic)": ("England", "Germany"),
    "🇺🇸 vs 🇲🇽  (Host derby)": ("United States", "Mexico"),
    "🇯🇵 vs 🇰🇷  (Asian rivalry)": ("Japan", "South Korea"),
    "🇲🇦 vs 🇸🇳  (African powers)": ("Morocco", "Senegal"),
}


def quick_pick(mu):
    return QUICK_PAIRS.get(mu, ("Spain", "Argentina"))


# Pre-render static elements
HERO_HTML = render_hero()
PODIUM_HTML = render_podium()
LEGEND_HTML = render_confed_legend()
STAGE_FIG = fig_stage_heatmap()
GROUPS_FIG = fig_group_visualizer()
OUTRIGHT_FIG = fig_outright_bar(20)
INIT_BRACKET, INIT_GROUPS = run_bracket(42)
INIT_SQUAD = render_team_squad("Spain", "All")


def render_outright(top_n): return fig_outright_bar(int(top_n))


# ============================== App ==============================
CSS = """
.gradio-container { max-width: 1300px !important; background: #fefcf7 !important; }
button.primary { background: linear-gradient(135deg, #5b8ec4, #3d6c9e) !important;
                 box-shadow: 0 2px 8px rgba(91,142,196,0.3) !important; }
.tab-nav button { font-weight: 600 !important; }
"""

with gr.Blocks(title="2026 WC Predictor", css=CSS,
               theme=gr.themes.Soft(primary_hue="blue", secondary_hue="amber", neutral_hue="stone")) as app:
    gr.HTML(HERO_HTML)

    with gr.Tabs():
        # ============ Tab 1 — Match Predictor
        with gr.Tab("📊 Match Predictor"):
            gr.Markdown("**두 팀을 골라 매치업 예측을 확인하세요** — 승무패 확률, 예상 스코어, 점수 분포, 주요 피처 분해")
            with gr.Row():
                quick = gr.Radio(choices=list(QUICK_PAIRS.keys()), label="🚀 빠른 선택", scale=4)
                neutral = gr.Checkbox(value=True, label="중립경기", scale=1)
                btn = gr.Button("⚽ 예측", variant="primary", size="lg", scale=1)
            with gr.Row():
                team_a = gr.Dropdown(WC_TEAMS, value="Spain", label="Team A")
                team_b = gr.Dropdown(WC_TEAMS, value="Argentina", label="Team B")

            summary_html = gr.HTML()
            with gr.Row():
                donut_plot = gr.Plot(label="")
                top_scores_plot = gr.Plot(label="")
            heatmap_plot = gr.Plot(label="")
            why_html = gr.HTML()

            quick.change(quick_pick, inputs=[quick], outputs=[team_a, team_b])
            btn.click(predict_match, inputs=[team_a, team_b, neutral],
                      outputs=[summary_html, donut_plot, heatmap_plot, top_scores_plot, why_html])
            app.load(predict_match, inputs=[team_a, team_b, neutral],
                     outputs=[summary_html, donut_plot, heatmap_plot, top_scores_plot, why_html])

        # ============ Tab 2 — Tournament Outlook
        with gr.Tab("🏆 Tournament Outlook"):
            gr.HTML(PODIUM_HTML)
            gr.HTML(LEGEND_HTML)
            gr.Markdown("### 우승 확률 Top N")
            top_n = gr.Slider(5, 48, value=20, step=1, label="Top N")
            outright_plot = gr.Plot(value=OUTRIGHT_FIG, label="")
            top_n.change(render_outright, inputs=[top_n], outputs=[outright_plot])

            gr.Markdown("### 48팀 단계별 진출 확률 히트맵")
            gr.Plot(value=STAGE_FIG, label="")

        # ============ Tab 3 — Group Stage
        with gr.Tab("🌍 Group Stage"):
            gr.HTML(LEGEND_HTML)
            gr.Markdown("### 12개 그룹 진출 확률")
            gr.Plot(value=GROUPS_FIG, label="")

        # ============ Tab 4 — Player Cards
        with gr.Tab("👤 Player Cards"):
            gr.Markdown("### FIFA / FM 스타일 선수 카드 — **⚔️ Attack 4 · 🛡️ Defense 4 · 💪 Physical 4** 스탯")
            gr.Markdown("*골키퍼는 자동으로 🥅 Distribution / ✋ Saves로 대체*")
            with gr.Row():
                squad_team = gr.Dropdown(WC_TEAMS, value="Spain", label="국가", scale=2)
                pos_filter = gr.Radio(["All", "GK", "DEF", "MID", "ATT"], value="All", label="포지션 필터", scale=3)
            squad_html = gr.HTML(value=INIT_SQUAD)
            squad_team.change(render_team_squad, inputs=[squad_team, pos_filter], outputs=[squad_html])
            pos_filter.change(render_team_squad, inputs=[squad_team, pos_filter], outputs=[squad_html])

        # ============ Tab 5 — Bracket
        with gr.Tab("🎲 Bracket Simulator"):
            gr.Markdown("### 단일 토너먼트 시뮬레이션 — Monte Carlo 한 샘플")
            gr.Markdown("seed를 바꾸면 다른 시나리오. 결과는 실제 다양한 가능성 중 *하나*.")
            with gr.Row():
                seed_in = gr.Slider(1, 1000, value=42, step=1, label="Random seed", scale=4)
                bracket_btn = gr.Button("🎲 새 시나리오 생성", variant="primary", scale=1)
            bracket_html = gr.HTML(value=INIT_BRACKET)
            gr.Markdown("### Group Stage 결과")
            groups_html = gr.HTML(value=INIT_GROUPS)
            bracket_btn.click(run_bracket, inputs=[seed_in], outputs=[bracket_html, groups_html])

        # ============ Tab 6 — About
        with gr.Tab("ℹ️ About"):
            gr.Markdown(f"""
## 모델 발전 history

| Phase | 시도 | 결과 |
|-------|------|------|
| 1~6 | 데이터 + 베이스라인 v1 (ELO only) | 53.5% |
| 7 | Recent Form (10경기 GD/CS) | ✓ **+1.6pp → v2.9** |
| 12 | ClubElo 휴리스틱 보정 | ✓ → v3.0 |
| 13 | Dixon-Coles τ | ✓ → **v3.5 (final)** |
| 8, 9, 10, 11, 14, 15, 16 | 다양한 시도 | ✗ 모두 효과 없음 |

## 핵심 발견
- **ELO는 압도적 단일 피처** — 25년 누적 정보
- **새로운 직교 정보만 효과** — Form, ClubElo만 유의미
- **데이터 천장 ~55%** — 공개 데이터로 도달 가능 한계
- **음의 결과의 가치** — 8개 phase 실패 검증

## 데이터 출처 + 시점
- 국제경기 결과: Kaggle (1872~2026-03), 49,215경기
- 선수 능력치: Kaggle **FIFA 23 (2022-23 시즌)** — 일부 선수의 소속 클럽은 최신과 다를 수 있음
- WC2026 조 편성/일정: Wikipedia (2025-12 추첨 결과)
- 클럽 ELO: ClubElo.com, 2026-05-10 기준
""")

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False, show_error=True)
