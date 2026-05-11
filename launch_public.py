"""Launch Gradio with public share URL. Chdir to user home to avoid OneDrive permission issues."""
import os, time
from pathlib import Path

# Save script's own directory before chdir
SCRIPT_DIR = Path(__file__).parent.resolve()
SHARE_FILE = SCRIPT_DIR / "share_url.txt"

# Chdir to user home — Gradio's .gradio cache directory creation blocked under OneDrive
os.chdir(os.path.expanduser("~"))

# Import after chdir
from gradio_app import app

# Launch — returned tuple shape can vary by Gradio version
result = app.launch(
    share=True, server_name="0.0.0.0", server_port=7860,
    inbrowser=False, show_error=True, prevent_thread_lock=True,
)
if isinstance(result, tuple):
    local_url = result[0]
    share_url = result[1] if len(result) > 1 else None
else:
    local_url = getattr(app, "local_url", None)
    share_url = getattr(app, "share_url", None)

SHARE_FILE.write_text(f"LOCAL: {local_url}\nSHARE: {share_url}\n", encoding="utf-8")
print(f"\n{'='*60}", flush=True)
print(f"  🌐 PUBLIC URL: {share_url}", flush=True)
print(f"  🏠 LOCAL URL:  {local_url}", flush=True)
print(f"{'='*60}\n", flush=True)
print(f"Written to: {SHARE_FILE}", flush=True)

try:
    while True: time.sleep(60)
except KeyboardInterrupt:
    pass
