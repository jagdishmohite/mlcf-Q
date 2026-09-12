import sys
from mlcfq.evaluation.capture_harness import capture_external_client_session
from mlcfq.extraction import extract_sessions_from_pcap

pcap, keylog = capture_external_client_session(
    binary_path=sys.executable,
    binary_args=[
        "../aioquic-src/examples/http3_client.py",
        "--secrets-log", "captures/aioquic-1.keylog",
    ],
    url="https://cloudflare-quic.com",
    out_pcap="captures/aioquic-1.pcapng",
    keylog_path="captures/aioquic-1.keylog",
    interface="Wi-Fi",
    timeout_s=30.0,
)

print(f"Captured: {pcap}")
sessions = extract_sessions_from_pcap(pcap, keylog_path=keylog)
print(f"Extracted {len(sessions)} QUIC session(s)")
for s in sessions:
    print(s.session_id, s.handshake.alpn, s.handshake.pq_key_share_offered, s.http3.settings)
