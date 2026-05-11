"""Deploy WC2026 predictor to Hugging Face Spaces."""
import os, sys, io, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
from huggingface_hub import HfApi, create_repo

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    sys.exit("ERROR: set HF_TOKEN env var. Example: $env:HF_TOKEN='hf_xxx'")
USERNAME = "67JM89"
SPACE_NAME = "wc2026-predictor"
REPO_ID = f"{USERNAME}/{SPACE_NAME}"

SCRIPT_DIR = Path(__file__).parent

# Files to include (whitelist approach for cleanliness)
INCLUDE_FILES = [
    # Entry + UI
    "app.py", "gradio_app.py", "wc_predictor.py", "i18n.py",
    # Config
    "requirements.txt",
    # Data (runtime needs this)
    "data/soccer.db",
]
# README will be uploaded separately from README_HF.md

api = HfApi(token=TOKEN)

print(f"=== Creating Space: {REPO_ID} ===")
try:
    create_repo(
        repo_id=REPO_ID, token=TOKEN, repo_type="space",
        space_sdk="gradio", exist_ok=True, private=False,
    )
    print(f"  ✓ Space exists or created")
except Exception as e:
    print(f"  ! Note: {e}")

print(f"\n=== Uploading files ===")
# Upload README first (with YAML frontmatter)
print(f"  → README.md (from README_HF.md)")
api.upload_file(
    path_or_fileobj=str(SCRIPT_DIR / "README_HF.md"),
    path_in_repo="README.md",
    repo_id=REPO_ID, repo_type="space", token=TOKEN,
)

for rel in INCLUDE_FILES:
    src = SCRIPT_DIR / rel
    if not src.exists():
        print(f"  ✗ skip (not found): {rel}")
        continue
    size_mb = src.stat().st_size / 1024 / 1024
    print(f"  → {rel}  ({size_mb:.1f} MB)")
    api.upload_file(
        path_or_fileobj=str(src), path_in_repo=rel,
        repo_id=REPO_ID, repo_type="space", token=TOKEN,
    )

print(f"\n=== Done ===")
print(f"🌐 Space URL: https://huggingface.co/spaces/{REPO_ID}")
print(f"   첫 빌드는 1-3분 소요. 'Building' → 'Running' 되면 접속 가능.")
