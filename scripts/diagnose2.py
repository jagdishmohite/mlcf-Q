import subprocess

# 1. Run the exact handshake command directly, see raw output
handshake_cmd = [
    "tshark", "-r", "captures/test-1.pcapng",
    "-Y", "tls.handshake.extensions_alpn_str",
    "-T", "fields",
    "-E", "separator=\t", "-E", "occurrence=a", "-E", "aggregator=,",
    "-e", "quic.dcid",
    "-e", "tls.handshake.ciphersuite",
    "-e", "tls.handshake.extension.type",
    "-e", "tls.handshake.extensions_supported_group",
    "-e", "tls.handshake.extensions_alpn_str",
]
result = subprocess.run(handshake_cmd, capture_output=True, text=True)
print("=== handshake command ===")
print("return code:", result.returncode)
print("stdout repr (first 2000 chars):")
print(repr(result.stdout[:2000]))
print("stderr:", result.stderr[:500])

# 2. Find the real HTTP/3 settings field names
print("\n=== searching for real http3 settings field names ===")
result2 = subprocess.run(
    ["tshark", "-r", "captures/test-1.pcapng", "-Y", "http3", "-T", "json", "-c", "3"],
    capture_output=True, text=True,
)
import json
try:
    packets = json.loads(result2.stdout)
    for pkt in packets:
        layers = pkt.get("_source", {}).get("layers", {})
        if "http3" in layers:
            print(json.dumps(layers["http3"], indent=2)[:3000])
            break
except json.JSONDecodeError:
    print("raw stdout:", result2.stdout[:1000])
    print("stderr:", result2.stderr[:500])
    