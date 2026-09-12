import subprocess

result = subprocess.run(
    [
        "tshark", "-r", "captures/test-1.pcapng",
        "-o", "tls.keylog_file:captures/test-1.keylog",
        "-Y", "http3.settings.id",
        "-T", "fields",
        "-E", "separator=\t", "-E", "occurrence=a", "-E", "aggregator=,",
        "-e", "quic.dcid",
        "-e", "http3.settings.id",
        "-e", "http3.settings.value",
    ],
    capture_output=True, text=True,
)
print("return code:", result.returncode)
print("stderr:", result.stderr[:500])
print("raw stdout:")
print(result.stdout)
