"""
One-click data download script for 2026 WC prediction model.

Prereq:
  1. pip install kaggle pandas requests beautifulsoup4 lxml
  2. ~/.kaggle/kaggle.json with API token
"""

import os
import sys
import io
from pathlib import Path

# Force stdout to UTF-8 on Windows (cp949 console can't print non-ASCII)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RAW = DATA / "raw"
DATA.mkdir(exist_ok=True)
RAW.mkdir(exist_ok=True)


def step(msg):
    print(f"\n{'='*60}\n>> {msg}\n{'='*60}")


# -----------------------------------------------------------------
# 1. Kaggle 데이터셋 다운로드
# -----------------------------------------------------------------
def download_kaggle():
    step("1/3  Kaggle datasets")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[FAIL] kaggle package missing. Run: pip install kaggle")
        sys.exit(1)

    api = KaggleApi()
    try:
        api.authenticate()
        print("Auth: OK")
    except Exception as e:
        print(f"[FAIL] Kaggle auth: {e}")
        print("       Check ~/.kaggle/kaggle.json")
        sys.exit(1)

    datasets = [
        # FIFA 23 player ratings (Pace/Shooting/Passing/Dribbling/Defending/Physical)
        ("stefanoleone992/fifa-23-complete-player-dataset", "fifa23"),
        # FIFA 22 (backup / additional season coverage)
        ("stefanoleone992/fifa-22-complete-player-dataset", "fifa22"),
        # International match results 1872-now (for team ELO + backtest)
        ("martj42/international-football-results-from-1872-to-2017", "intl_results"),
    ]

    for slug, folder in datasets:
        out = RAW / folder
        out.mkdir(exist_ok=True)
        print(f"  -> {slug}")
        try:
            api.dataset_download_files(slug, path=str(out), unzip=True, quiet=True)
            print(f"     OK -> {out}")
        except Exception as e:
            print(f"     [FAIL] {type(e).__name__}: {str(e)[:100]}")


# -----------------------------------------------------------------
# 2. Wikipedia 에서 2026 월드컵 정보 스크랩
# -----------------------------------------------------------------
def scrape_wc2026():
    step("2/3  2026 World Cup tables (Wikipedia)")
    import requests
    from io import StringIO

    urls = [
        ("main",  "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup"),
        ("qual",  "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_qualification"),
    ]
    out_dir = RAW / "wc2026"
    out_dir.mkdir(exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0; data-collection)",
    }

    total = 0
    for tag, url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            tables = pd.read_html(StringIO(r.text))
            tables_sorted = sorted(tables, key=lambda t: t.size, reverse=True)
            for i, t in enumerate(tables_sorted[:10]):
                t.to_csv(out_dir / f"{tag}_t{i:02d}.csv", index=False, encoding="utf-8-sig")
                total += 1
            print(f"  OK {url}  ({len(tables)} tables, top10 saved)")
        except Exception as e:
            print(f"  [FAIL] {url}: {type(e).__name__}: {str(e)[:80]}")
    print(f"  Saved {total} CSVs to {out_dir}")


# -----------------------------------------------------------------
# 3. 받은 파일 확인 + 정리
# -----------------------------------------------------------------
def summarize():
    step("3/3  Summary")

    print("\nRAW files (data/raw/)")
    for path in sorted(RAW.rglob("*")):
        if path.is_file():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"  {path.relative_to(RAW)}  ({size_mb:.1f} MB)")

    print("\nCSV preview")
    for csv in sorted(RAW.rglob("*.csv")):
        try:
            df_full = pd.read_csv(csv, low_memory=False)
            rel = csv.relative_to(RAW)
            cols_show = list(df_full.columns)[:6]
            print(f"  {rel}")
            print(f"    rows={len(df_full):,}  cols={len(df_full.columns)}  first6={cols_show}")
        except Exception as e:
            print(f"  {csv.relative_to(RAW)}: read failed ({type(e).__name__})")


if __name__ == "__main__":
    download_kaggle()
    scrape_wc2026()
    summarize()

    print(f"\n{'='*60}")
    print("DONE. Next: inspect data/raw/ then build unified SQLite.")
    print(f"{'='*60}")
