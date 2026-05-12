"""
Scrape Transfermarkt national-team squads for all 48 WC2026 teams.

Output: SQLite table `tm_squads` with columns:
  team, tm_player_id, name, age, position, current_club, market_value_eur, is_injured

Run once; results cached. Idempotent.
"""
import sys, io, sqlite3, re, time, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
import requests
from bs4 import BeautifulSoup

DB = Path(__file__).parent / "data" / "soccer.db"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
HEADERS = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}

# Cached IDs (filled by search on first run)
TM_ID_CACHE = Path(__file__).parent / "data" / "tm_nation_ids.json"

# WC2026 team names (canonical, English)
WC_TEAMS = [
    "Algeria", "Argentina", "Australia", "Austria",
    "Belgium", "Bosnia and Herzegovina", "Brazil",
    "Canada", "Cape Verde", "Colombia", "Croatia",
    "Curaçao", "Czech Republic", "DR Congo",
    "Ecuador", "Egypt", "England",
    "France", "Germany", "Ghana", "Haiti",
    "Iran", "Iraq", "Ivory Coast",
    "Japan", "Jordan", "Mexico", "Morocco",
    "Netherlands", "New Zealand", "Norway",
    "Panama", "Paraguay", "Portugal",
    "Qatar", "Saudi Arabia", "Scotland",
    "Senegal", "South Africa", "South Korea",
    "Spain", "Sweden", "Switzerland",
    "Tunisia", "Turkey", "United States",
    "Uruguay", "Uzbekistan",
]

# Search aliases (TM uses non-English names for some)
TM_SEARCH_NAME = {
    "South Korea": "Korea, South",
    "Ivory Coast": "Cote d'Ivoire",
    "DR Congo": "DR Kongo",
    "Czech Republic": "Czech Republic",
    "Curaçao": "Curacao",
    "United States": "United States",
}


def search_nation_id(team: str, session: requests.Session) -> int | None:
    """Search Transfermarkt for a national team ID. Returns first verein match
    whose page contains 'Nationalmannschaft' indicator."""
    query = TM_SEARCH_NAME.get(team, team)
    url = "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche"
    r = session.get(url, headers=HEADERS, params={"query": query}, timeout=12)
    if r.status_code != 200:
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    # Find "Klubs" (clubs) table — national teams are listed under clubs
    for link in soup.select('a[href*="/startseite/verein/"]'):
        href = link.get("href", "")
        m = re.search(r"/verein/(\d+)", href)
        if not m: continue
        text = link.get_text(strip=True)
        if text.lower() == query.lower() or text.lower() == team.lower():
            return int(m.group(1))
    # Fallback: first /verein/ link in results
    for link in soup.select('a[href*="/startseite/verein/"]'):
        m = re.search(r"/verein/(\d+)", link.get("href", ""))
        if m:
            return int(m.group(1))
    return None


def load_or_search_ids(session: requests.Session) -> dict[str, int]:
    if TM_ID_CACHE.exists():
        cached = json.loads(TM_ID_CACHE.read_text(encoding="utf-8"))
        print(f"  Loaded {len(cached)} cached IDs from {TM_ID_CACHE.name}")
        return cached
    print("=== Searching TM for national team IDs (cached for next run) ===\n")
    ids = {}
    for i, team in enumerate(WC_TEAMS, 1):
        tid = search_nation_id(team, session)
        if tid:
            ids[team] = tid
            print(f"  [{i:2d}/48] {team:25s} → {tid}")
        else:
            print(f"  [{i:2d}/48] {team:25s} → NOT FOUND")
        time.sleep(0.4)
    TM_ID_CACHE.write_text(json.dumps(ids, indent=2), encoding="utf-8")
    print(f"\n  Cached to {TM_ID_CACHE}")
    return ids

# Slug for URL — TM uses team-native slugs but accepts /x/kader/verein/{id} as canonical
def squad_url(tid: int, season: int = 2024) -> str:
    return f"https://www.transfermarkt.com/x/kader/verein/{tid}/saison_id/{season}"


def parse_value(s: str) -> float | None:
    """'€80.00m' -> 80_000_000.0; '€450k' -> 450_000.0; '-' -> None"""
    if not s or s == "-": return None
    s = s.replace("€", "").strip()
    if s.endswith("m"):
        return float(s[:-1]) * 1_000_000
    if s.endswith("k"):
        return float(s[:-1]) * 1_000
    try: return float(s)
    except: return None


def fetch_squad(team_name: str, tid: int, session: requests.Session) -> list[dict]:
    """Returns list of player dicts."""
    url = squad_url(tid)
    r = session.get(url, headers=HEADERS, timeout=15)
    if r.status_code != 200:
        print(f"  ✗ {team_name}: HTTP {r.status_code}")
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    rows = soup.select("table.items > tbody > tr.odd, table.items > tbody > tr.even")
    players = []
    for row in rows:
        try:
            # Player ID + name
            link = row.select_one('a[href*="/profil/spieler/"]')
            if not link: continue
            href = link.get("href", "")
            m = re.search(r"/spieler/(\d+)", href)
            if not m: continue
            pid = int(m.group(1))
            name = link.get_text(strip=True)

            # Jersey number / position type from first td
            first_td = row.select_one("td.zentriert.rueckennummer")
            pos_type = first_td.get("title", "") if first_td else ""

            # All td cells (string content)
            tds = row.find_all("td", recursive=False)
            # Age is in tds[2] usually: "01/02/1995 (29)"
            age = None
            for td in tds:
                txt = td.get_text(" ", strip=True)
                ma = re.search(r"\((\d{2})\)", txt)
                if ma:
                    age = int(ma.group(1))
                    break

            # Position (e.g., 'Centre-Forward') — appears inside inline table
            inline = row.select_one("table.inline-table")
            pos = None
            if inline:
                pos_td = inline.select("td")
                if len(pos_td) >= 2:
                    pos_text = pos_td[-1].get_text(strip=True)
                    if pos_text and pos_text != name:
                        pos = pos_text

            # Current club — find <img class="tiny_wappen"> or link to /verein/
            club = None
            club_link = row.select_one('a[href*="/verein/"][title]')
            if club_link:
                club = club_link.get("title", "").strip() or None

            # Market value — last column with text starting with €
            mv_eur = None
            for td in tds[::-1]:
                txt = td.get_text(strip=True)
                if txt.startswith("€"):
                    mv_eur = parse_value(txt)
                    break

            # Injury marker — TM has a "verletzt" CSS class on tooltip
            injured = bool(row.select_one('span.verletzt-table, span[title*="njur"], span[title*="erletzt"]'))

            players.append({
                "team": team_name, "tm_player_id": pid, "name": name,
                "age": age, "position": pos, "pos_type": pos_type,
                "current_club": club, "market_value_eur": mv_eur,
                "is_injured": 1 if injured else 0,
            })
        except Exception as e:
            print(f"  ⚠ row parse error: {e}")
            continue
    return players


def main():
    session = requests.Session()
    ids = load_or_search_ids(session)

    print(f"\n=== Scraping {len(ids)} national-team squads ===\n")
    all_players = []
    for i, (team, tid) in enumerate(sorted(ids.items()), 1):
        players = fetch_squad(team, tid, session)
        all_players.extend(players)
        print(f"  [{i:2d}/{len(ids)}] {team:25s} (id={tid}) → {len(players):2d} players")
        time.sleep(0.4)  # politeness
    print(f"\n=== Total: {len(all_players):,} player-rows ===")

    # Write to SQLite
    conn = sqlite3.connect(DB)
    conn.execute("DROP TABLE IF EXISTS tm_squads")
    conn.execute("""
        CREATE TABLE tm_squads (
            team TEXT, tm_player_id INTEGER, name TEXT,
            age INTEGER, position TEXT, pos_type TEXT,
            current_club TEXT, market_value_eur REAL,
            is_injured INTEGER,
            PRIMARY KEY (team, tm_player_id)
        )
    """)
    conn.executemany("""
        INSERT OR REPLACE INTO tm_squads
        (team, tm_player_id, name, age, position, pos_type, current_club, market_value_eur, is_injured)
        VALUES (:team, :tm_player_id, :name, :age, :position, :pos_type, :current_club, :market_value_eur, :is_injured)
    """, all_players)
    conn.commit()

    # Summary
    print("\n=== Summary by team ===")
    import pandas as pd
    df = pd.read_sql("""
        SELECT team, COUNT(*) n, SUM(market_value_eur)/1e6 total_mv_m, SUM(is_injured) injured
        FROM tm_squads GROUP BY team ORDER BY total_mv_m DESC
    """, conn)
    print(df.to_string())
    conn.close()


if __name__ == "__main__":
    main()
