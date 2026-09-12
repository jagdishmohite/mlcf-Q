"""Orchestrate real QUIC/HTTP-3 traffic generation and capture (paper §6-7).

This is the one module in the evaluation package that genuinely cannot be
exercised in a sandboxed environment: it drives real browsers, runs real
library-client binaries, and shells out to `tshark` against a real network
interface. Everything here is written to be run on your own machine.

Prerequisites (none of these are installed by `pip install -e .`):

  - `tshark` on PATH (Wireshark CLI) -- same requirement as
    `mlcfq.extraction.pcap_extractor`.
  - Playwright with browser binaries installed, for the browser capture
    functions: `pip install playwright && playwright install chromium firefox`.
  - `aioquic` installed (`pip install aioquic`) for the aioquic capture
    function -- it's a pure-Python QUIC/HTTP-3 stack, so no separate binary
    build is needed.
  - `quic-go`'s example client and `ngtcp2`'s example client built
    separately (both are non-Python; see their own build instructions) if
    you want those library stacks in your dataset. Point
    `capture_external_client_session` at the built binary.
  - A `SSLKEYLOGFILE`-capable client for HTTP/3 frame-level detail (paper
    §6) -- Chromium/Firefox and aioquic all support the standard
    `SSLKEYLOGFILE` environment variable; set it before capture and pass
    the same path to `extract_sessions_from_pcap`.

Every capture function here returns the raw `(pcap_path, keylog_path)` it
produced; feed those into `mlcfq.extraction.extract_sessions_from_pcap` to
get `Session` objects, then wrap in `LabeledSession` with `tier="genuine"`
and the appropriate `ground_truth_class` yourself -- labeling is a
one-line wrap, shown in each function's docstring, since only you know
what you actually ran.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


class CaptureError(RuntimeError):
    """Raised when a capture subprocess fails or required tools are missing."""


def _check_tool(name: str) -> None:
    from shutil import which

    if which(name) is None:
        raise CaptureError(
            f"'{name}' not found on PATH. See this module's docstring for prerequisites."
        )


def start_tshark_capture(interface: str, out_pcap: str, capture_filter: str = "udp port 443") -> subprocess.Popen:
    """Start a background tshark capture. Call `stop_tshark_capture` when done.

    Uses a BPF *capture* filter (`-f`), not a display filter (`-Y`) --
    tshark does not allow combining `-Y` with `-w` (writing raw packets to
    a file), since `-Y` requires per-packet analysis that conflicts with a
    straight capture-to-disk write. The default `"udp port 443"` narrows
    to the conventional HTTPS/QUIC port (cutting out unrelated UDP noise
    like DNS on port 53) without needing protocol-level parsing at capture
    time. QUIC-specific filtering happens afterward, when
    `extract_sessions_from_pcap` reads the file back with its own
    `display_filter="quic"` default -- that's the correct point for a
    display filter, since it's applied to an already-written file rather
    than a live write stream.

    Requires permission to capture on `interface` (on Linux, either run as
    root or grant the capability: `sudo setcap cap_net_raw,cap_net_admin+eip $(which tshark)`;
    on Windows, run your terminal as Administrator).

    Raises `CaptureError` immediately if tshark exits within the startup
    check window (a bad interface name, invalid filter syntax, and
    permission errors all show up this way) rather than continuing on with
    a capture process that already failed -- letting a browser automation
    step run for several seconds against a tshark that's already dead is
    exactly how a capture silently produces an empty file.
    """
    _check_tool("tshark")
    Path(out_pcap).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["tshark", "-i", interface, "-f", capture_filter, "-w", out_pcap]
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

    # tshark prints "Capturing on '<name>'" to stderr once it actually
    # starts listening; give it a moment and check it didn't die on
    # startup (bad interface, permissions, bad filter syntax all exit
    # immediately with a non-zero code).
    time.sleep(0.5)
    if process.poll() is not None:
        stderr_output = process.stderr.read() if process.stderr else ""
        raise CaptureError(
            f"tshark exited immediately (exit code {process.returncode}); "
            f"capture never started. tshark said:\n{stderr_output}"
        )
    return process


def stop_tshark_capture(process: subprocess.Popen, grace_period_s: float = 1.0, out_pcap: str | None = None) -> None:
    """Stop a capture started with `start_tshark_capture`, letting tshark
    flush its output file cleanly.

    If `out_pcap` is given, verifies the file actually exists and is
    non-empty afterward, raising `CaptureError` if not -- catching the
    case where tshark was running but captured zero matching packets
    (e.g. the capture filter matched nothing, or the traffic never
    actually went out over the interface being watched).
    """
    time.sleep(grace_period_s)  # let in-flight packets land before stopping
    process.terminate()
    stderr_output = ""
    try:
        _, stderr_output = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()

    if out_pcap is not None:
        path = Path(out_pcap)
        if not path.exists() or path.stat().st_size == 0:
            detail = f"\ntshark said:\n{stderr_output}" if stderr_output else ""
            raise CaptureError(
                f"capture produced no output at {out_pcap} -- tshark ran but the file is "
                f"missing or empty. Common causes: no packets matched the capture filter "
                f"(nothing on that port during the capture window), or the wrong interface "
                f"was specified.{detail}"
            )


def capture_browser_session(
    browser: str,
    url: str,
    out_pcap: str,
    keylog_path: str,
    interface: str = "lo",
    headless: bool = True,
    wait_s: float = 5.0,
) -> tuple[str, str]:
    """Capture one QUIC/HTTP-3 session from a real browser via Playwright.

    `browser` is "chromium" or "firefox". Requires Playwright + browser
    binaries installed (see module docstring).

    Chromium and Firefox use different mechanisms for TLS key logging, and
    only one of them was empirically confirmed to work reliably through
    Playwright:

    - **Chromium**: launched with `--ssl-key-log-file=<path>` as a
      command-line argument. An earlier version of this function set the
      `SSLKEYLOGFILE` *environment variable* instead (which Chromium is
      documented to also support) -- that approach was tested against a
      real capture and silently failed to produce any keylog output at
      all, with no error from either Playwright or Chromium. The
      command-line flag was tested the same way and worked. Environment
      variable propagation through Playwright's `env=` launch parameter is
      the suspected culprit, though the exact mechanism wasn't pinned down
      further once a working alternative was confirmed.
    - **Firefox**: uses the `SSLKEYLOGFILE` environment variable natively
      (built into NSS, not a Chromium-specific flag), so that's what's used
      here for Firefox specifically. This path has *not* been separately
      verified against a real capture the way the Chromium path was --
      if Firefox captures also come up with an empty keylog, the fix is
      likely the same class of problem (env propagation through
      Playwright), and the next thing to try is Firefox's `MOZ_LOG`-style
      preference-based configuration instead.

    Raises `CaptureError` if the keylog file doesn't exist or is empty
    after the browser closes, rather than silently returning a path with
    nothing useful in it.

    Label the result as, e.g.::

        sessions = extract_sessions_from_pcap(pcap_path, keylog_path)
        labeled = [
            LabeledSession(session=s, ground_truth_class="Chrome-QUIC", tier="genuine")
            for s in sessions
        ]
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise CaptureError(
            "playwright not installed: pip install playwright && playwright install"
        ) from exc

    keylog_abs_path = str(Path(keylog_path).resolve())
    Path(keylog_path).parent.mkdir(parents=True, exist_ok=True)
    capture_proc = start_tshark_capture(interface, out_pcap)
    try:
        with sync_playwright() as p:
            browser_type = getattr(p, browser)
            if browser == "chromium":
                # Confirmed working: command-line flag, not env var.
                instance = browser_type.launch(
                    headless=headless, args=[f"--ssl-key-log-file={keylog_abs_path}"]
                )
            else:
                # Firefox: SSLKEYLOGFILE env var (NSS-native). Not yet
                # separately verified against a real capture -- see
                # docstring.
                launch_env = {**os.environ, "SSLKEYLOGFILE": keylog_abs_path}
                instance = browser_type.launch(headless=headless, env=launch_env)
            page = instance.new_page()
            page.goto(url, wait_until="networkidle")
            time.sleep(wait_s)  # let 0-RTT/connection-reuse behavior settle
            instance.close()
    finally:
        stop_tshark_capture(capture_proc, out_pcap=out_pcap)

    keylog_file = Path(keylog_path)
    if not keylog_file.exists() or keylog_file.stat().st_size == 0:
        raise CaptureError(
            f"keylog file at {keylog_path} was not created or is empty -- TLS/QUIC "
            f"traffic was captured but cannot be decrypted for HTTP/3-layer feature "
            f"extraction. For Chromium this usually means the --ssl-key-log-file flag "
            f"isn't taking effect; for Firefox it usually means the SSLKEYLOGFILE env "
            f"var isn't propagating through Playwright's launch() call."
        )


    return out_pcap, keylog_path


def capture_aioquic_session(
    url: str,
    out_pcap: str,
    keylog_path: str,
    interface: str = "lo",
    aioquic_client_args: list[str] | None = None,
) -> tuple[str, str]:
    """Capture one session from aioquic's reference HTTP/3 client.

    Requires `aioquic` installed and its `examples/http3_client.py` (from
    the aioquic source tree) available -- pass its path as the first
    element of `aioquic_client_args`, e.g.
    `["python3", "/path/to/aioquic/examples/http3_client.py", "--ca-certs", "..."]`.
    aioquic honors `SSLKEYLOGFILE` natively.

    Label the result as, e.g.::

        LabeledSession(session=s, ground_truth_class="aioquic-library-client", tier="genuine")
    """
    Path(keylog_path).parent.mkdir(parents=True, exist_ok=True)
    capture_proc = start_tshark_capture(interface, out_pcap)
    try:
        args = aioquic_client_args or ["python3", "-m", "aioquic.examples.http3_client"]
        env = {**os.environ, "SSLKEYLOGFILE": keylog_path}
        result = subprocess.run(args + [url], env=env, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise CaptureError(f"aioquic client failed: {result.stderr}")
    finally:
        stop_tshark_capture(capture_proc, out_pcap=out_pcap)

    return out_pcap, keylog_path


def capture_external_client_session(
    binary_path: str,
    binary_args: list[str],
    url: str,
    out_pcap: str,
    keylog_path: str | None = None,
    interface: str = "lo",
    timeout_s: float = 30.0,
) -> tuple[str, str | None]:
    """Capture one session from a pre-built external QUIC client binary
    (e.g. quic-go's or ngtcp2's example client).

    You must build these yourself -- neither ships a Python package. Point
    `binary_path` at the built executable and `binary_args` at whatever
    flags it needs (consult that project's own examples/README). If the
    binary supports `SSLKEYLOGFILE`, set `keylog_path`; not all example
    clients do, in which case pass `None` and expect sparser `x^(A)`
    (HTTP/3 frame-level) features on extraction, per the honest caveat in
    paper §6.

    Label the result as, e.g.::

        LabeledSession(session=s, ground_truth_class="quic-go-library-client", tier="genuine")
    """
    capture_proc = start_tshark_capture(interface, out_pcap)
    try:
        env = dict(os.environ)
        if keylog_path:
            Path(keylog_path).parent.mkdir(parents=True, exist_ok=True)
            env["SSLKEYLOGFILE"] = keylog_path
        result = subprocess.run(
            [binary_path] + binary_args + [url],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if result.returncode != 0:
            raise CaptureError(f"{binary_path} failed: {result.stderr}")
    finally:
        stop_tshark_capture(capture_proc, out_pcap=out_pcap)

    return out_pcap, keylog_path


def trigger_connection_migration(interface: str, pause_s: float = 2.0) -> None:
    """Best-effort connection-migration trigger: briefly bring an interface
    down and back up mid-capture, forcing the client to migrate to a new
    path (or fail over) if it's mid-connection.

    This is genuinely best-effort and platform-dependent -- on a real
    mobile device, migration is triggered by an actual Wi-Fi/cellular
    handover, which this can't simulate from a Linux capture box. Use this
    only as a rough approximation, and prefer real network handovers on a
    mobile test rig if your dataset needs authentic migration events.
    Requires root/sudo.
    """
    subprocess.run(["sudo", "ip", "link", "set", interface, "down"], check=False)
    time.sleep(pause_s)
    subprocess.run(["sudo", "ip", "link", "set", interface, "up"], check=False)
