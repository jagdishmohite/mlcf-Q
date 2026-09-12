import pyshark

capture = pyshark.FileCapture(
    "captures/test-1.pcapng",
    display_filter="quic",
    use_json=True,
    include_raw=False,
)

for packet in capture:
    quic_layer = packet.quic
    all_fields = quic_layer._all_fields
    alpn_keys = [k for k in all_fields if "alpn" in k.lower()]
    tls_keys = [k for k in all_fields if "tls" in k.lower() or "handshake" in k.lower()]
    if alpn_keys or tls_keys:
        print("=== packet with handshake data ===")
        print("ALPN-related keys:", alpn_keys)
        for k in alpn_keys:
            print(f"  {k} = {all_fields[k]}")
        print("all TLS/handshake-related keys found:")
        for k in sorted(tls_keys):
            print(f"  {k} = {all_fields[k]}")
        break

capture.close()