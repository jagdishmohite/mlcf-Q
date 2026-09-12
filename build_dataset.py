# build_dataset.py (v2 -- scaled up)
#
# Captures many more repeats of Chromium and aioquic sessions than the
# first version, spread across two real HTTP/3 sites instead of hitting
# one repeatedly. Filters to sessions with real handshake data, builds
# A1/A2 adversarial partitions, saves everything as one labeled dataset.
#
# What changed from v1:
#   - REPEATS raised from 5 to 15 per (class, URL) combination
#   - Two target URLs instead of one, for site diversity and to reduce the
#     odds of one site's bot detection intermittently blocking repeated
#     headless automation (v1 saw several Chrome captures silently return
#     0 sessions -- plausibly triggered by hitting the same site
#     back-to-back many times)
#   - A short randomized delay between captures, same reasoning
#   - Each LabeledSession's source_meta now records which URL produced it,
#     so later analysis can check for URL-specific effects
#
# Honest scope note carried over from v1: every "genuine" Chromium session
# here is captured via Playwright automation, not organic human browsing --
# see v1's docstring for the full explanation of why that's a known,
# flagged gap relative to paper Section 4's A3 tier, not something this
# script tries to paper over.

import random
import sys
import time

from mlcfq.evaluation.adversarial import build_a1_a2_partitions
from mlcfq.evaluation.capture_harness import (
    CaptureError,
    capture_browser_session,
    capture_external_client_session,
)
from mlcfq.evaluation.dataset import LabeledSession, save_labeled_sessions, summarize
from mlcfq.extraction import extract_sessions_from_pcap

URLS = ["https://cloudflare-quic.com", "https://quic.nginx.org"]
INTERFACE = "Wi-Fi"  # your confirmed Wi-Fi interface number from `tshark -D`
REPEATS_PER_URL = 15

genuine_sessions: list[LabeledSession] = []


def capture_chrome(url: str, i: int) -> None:
    pcap = f"captures/dataset-chrome-{i}.pcapng"
    keylog = f"captures/dataset-chrome-{i}.keylog"
    try:
        capture_browser_session("chromium", url, pcap, keylog, interface=INTERFACE, headless=True)
        sessions = extract_sessions_from_pcap(pcap, keylog_path=keylog)
        with_handshake = [s for s in sessions if s.handshake.alpn]
        genuine_sessions.extend(
            LabeledSession(
                session=s,
                ground_truth_class="Chrome-QUIC",
                tier="genuine",
                source_meta={"captured_url": url},
            )
            for s in with_handshake
        )
        print(f"[chrome {i} @ {url}] {len(sessions)} extracted, {len(with_handshake)} kept")
    except CaptureError as e:
        print(f"[chrome {i} @ {url}] capture failed, skipping: {e}")


def capture_aioquic(url: str, i: int) -> None:
    pcap = f"captures/dataset-aioquic-{i}.pcapng"
    keylog = f"captures/dataset-aioquic-{i}.keylog"
    try:
        capture_external_client_session(
            binary_path=sys.executable,
            binary_args=["../aioquic-src/examples/http3_client.py", "--secrets-log", keylog],
            url=url,
            out_pcap=pcap,
            keylog_path=keylog,
            interface=INTERFACE,
        )
        sessions = extract_sessions_from_pcap(pcap, keylog_path=keylog)
        with_handshake = [s for s in sessions if s.handshake.alpn]
        genuine_sessions.extend(
            LabeledSession(
                session=s,
                ground_truth_class="aioquic-library-client",
                tier="genuine",
                source_meta={"captured_url": url},
            )
            for s in with_handshake
        )
        print(f"[aioquic {i} @ {url}] {len(sessions)} extracted, {len(with_handshake)} kept")
    except CaptureError as e:
        print(f"[aioquic {i} @ {url}] capture failed, skipping: {e}")


# --- Chromium captures ---
i = 0
for url in URLS:
    for _ in range(REPEATS_PER_URL):
        capture_chrome(url, i)
        i += 1
        time.sleep(random.uniform(2.0, 4.0))

# --- aioquic captures ---
i = 0
for url in URLS:
    for _ in range(REPEATS_PER_URL):
        capture_aioquic(url, i)
        i += 1
        time.sleep(random.uniform(1.0, 2.0))

print(f"\nTotal genuine sessions with real handshake data: {len(genuine_sessions)}")
chrome_count = sum(1 for s in genuine_sessions if s.ground_truth_class == "Chrome-QUIC")
aioquic_count = sum(1 for s in genuine_sessions if s.ground_truth_class == "aioquic-library-client")
print(f"  Chrome-QUIC: {chrome_count}")
print(f"  aioquic-library-client: {aioquic_count}")

# --- Build A1/A2 adversarial partitions from what we actually captured ---
a1_sessions, a2_sessions = [], []
try:
    a1_sessions, a2_sessions = build_a1_a2_partitions(
        genuine_sessions,
        target_classes=["Chrome-QUIC"],
        source_classes=["aioquic-library-client"],
    )
    print(f"Built {len(a1_sessions)} A1 and {len(a2_sessions)} A2 adversarial session(s)")
except ValueError as e:
    print(f"Could not build adversarial partitions (likely too few genuine sessions of one class): {e}")

# --- Save everything ---
all_sessions = genuine_sessions + a1_sessions + a2_sessions
save_labeled_sessions(all_sessions, "examples/real_dataset.json")
print(f"\nSaved {len(all_sessions)} total labeled sessions to examples/real_dataset.json")
print(summarize(all_sessions))
