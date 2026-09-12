from mlcfq.evaluation.capture_harness import capture_browser_session
from mlcfq.extraction import extract_sessions_from_pcap

pcap, keylog = capture_browser_session(
    browser="chromium",
    url="https://cloudflare-quic.com",
    out_pcap="captures/test-1.pcapng",
    keylog_path="captures/test-1.keylog",
    interface="Wi-Fi",
    headless=False,
)

print(f"Captured: {pcap}")
sessions = extract_sessions_from_pcap(pcap, keylog_path=keylog)
print(f"Extracted {len(sessions)} QUIC session(s)")
for s in sessions:
    print(s.session_id, s.handshake.alpn, s.transport.quic_version)