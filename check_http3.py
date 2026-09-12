from mlcfq.extraction import extract_sessions_from_pcap

sessions = extract_sessions_from_pcap(
    "captures/test-1.pcapng",
    keylog_path="captures/test-1.keylog",
)

for s in sessions:
    if s.handshake.alpn:
        print(s.session_id, s.handshake.alpn, s.http3.settings, s.http3.h3_negotiated)

        