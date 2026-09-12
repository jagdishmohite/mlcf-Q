import subprocess
import json

result = subprocess.run(
    ["tshark", "-r", "captures/test-1.pcapng", "-Y", "http3", "-T", "json"],
    capture_output=True, text=True,
)
print("return code:", result.returncode)
print("stderr:", result.stderr[:500])
print("stdout length:", len(result.stdout))

if result.stdout.strip():
    packets = json.loads(result.stdout)
    print(f"matched {len(packets)} packet(s) with an http3 layer")
    if packets:
        layers = packets[0].get("_source", {}).get("layers", {})
        print("top-level layer keys:", list(layers.keys()))
        if "http3" in layers:
            print(json.dumps(layers["http3"], indent=2)[:3000])
else:
    print("tshark matched ZERO packets for display filter 'http3'")
    