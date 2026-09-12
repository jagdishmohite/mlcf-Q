import time
from pathlib import Path
from playwright.sync_api import sync_playwright

keylog_path = str(Path("captures/manual-keylog-test2.keylog").resolve())
Path(keylog_path).parent.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(
        headless=False,
        args=[f"--ssl-key-log-file={keylog_path}"],
    )
    page = browser.new_page()
    page.goto("https://cloudflare-quic.com", wait_until="networkidle")
    time.sleep(5)
    browser.close()

if Path(keylog_path).exists():
    size = Path(keylog_path).stat().st_size
    print(f"SUCCESS: keylog file exists, {size} bytes")
else:
    print("FAILED: keylog file still was not created")