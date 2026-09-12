"""`mlcfq` command-line interface.

    mlcfq extract --pcap capture.pcapng [--keylog secrets.log] --out sessions.json
    mlcfq score   --session sessions.json --profiles profiles.json [--lambda 1.0]
    mlcfq cohort  --sessions sessions.json [--baseline baseline.json] [--threshold 1.5]
    mlcfq study   --dataset labeled_sessions.json --profiles profiles.json [--out report.md]
"""

from __future__ import annotations

import json

import click

from mlcfq.pipeline import load_profiles, load_sessions, result_to_dict, run_cohort_test, score_sessions


@click.group()
@click.version_option()
def main():
    """MLCF-Q: Multi-Layer Coherence Forensics for QUIC/HTTP-3 traffic attribution."""


@main.command()
@click.option("--pcap", required=True, type=click.Path(exists=True), help="Input pcap/pcapng file")
@click.option("--keylog", default=None, type=click.Path(exists=True), help="SSLKEYLOGFILE for HTTP/3 detail")
@click.option("--out", required=True, type=click.Path(), help="Output JSON path for extracted sessions")
def extract(pcap: str, keylog: str | None, out: str):
    """Decode a pcap into MLCF-Q Session records (requires tshark/pyshark)."""
    from mlcfq.extraction import extract_sessions_from_pcap

    sessions = extract_sessions_from_pcap(pcap, keylog_path=keylog)
    with open(out, "w") as f:
        json.dump([s.to_dict() for s in sessions], f, indent=2)
    click.echo(f"Extracted {len(sessions)} session(s) -> {out}")


@main.command()
@click.option("--session", "session_path", required=True, type=click.Path(exists=True))
@click.option("--profiles", "profiles_path", required=True, type=click.Path(exists=True))
@click.option("--lambda", "lam", default=1.0, type=float, help="Cross-layer penalty coefficient λ")
@click.option("--top", default=3, type=int, help="Number of ranked profiles to show per session")
def score(session_path: str, profiles_path: str, lam: float, top: int):
    """Score session(s) against candidate client-stack profiles (Instantiation R)."""
    sessions = load_sessions(session_path)
    profiles = load_profiles(profiles_path)
    results = score_sessions(sessions, profiles, lam=lam)

    for session_id, ranked in results.items():
        click.echo(f"\nsession {session_id}")
        for result in ranked[:top]:
            click.echo(f"  {json.dumps(result_to_dict(result))}")


@main.command()
@click.option("--sessions", "sessions_path", required=True, type=click.Path(exists=True))
@click.option("--baseline", "baseline_path", default=None, type=click.Path(exists=True))
@click.option("--threshold", default=1.5, type=float, help="Flag threshold on dispersion ratio")
@click.option("--min-cohort-size", default=3, type=int)
def cohort(sessions_path: str, baseline_path: str | None, threshold: float, min_cohort_size: int):
    """Run the cohort-level transport-parameter dispersion test (§5.3)."""
    sessions = load_sessions(sessions_path)
    baseline = json.loads(open(baseline_path).read()) if baseline_path else None

    results = run_cohort_test(
        sessions, baseline_by_key=baseline, flag_threshold=threshold, min_cohort_size=min_cohort_size
    )

    if not results:
        click.echo("No cohorts met the minimum size threshold.")
        return

    for r in results:
        flag = " [FLAGGED]" if r.flagged else ""
        click.echo(
            f"{r.handshake_key[:40]:40s} n={r.session_count:3d} "
            f"unique_sig={r.unique_signature_count:3d} "
            f"dispersion={r.normalized_dispersion:.3f}"
            + (f" ratio={r.dispersion_ratio:.2f}" if r.dispersion_ratio is not None else "")
            + flag
        )


@main.command()
@click.option("--dataset", "dataset_path", required=True, type=click.Path(exists=True), help="Labeled sessions JSON (mlcfq.evaluation.dataset format)")
@click.option("--profiles", "profiles_path", required=True, type=click.Path(exists=True))
@click.option("--out", default=None, type=click.Path(), help="Write Markdown report here (default: print to stdout)")
def study(dataset_path: str, profiles_path: str, out: str | None):
    """Run the RQ1/RQ2 evaluation study over a labeled dataset (§7).

    The dataset must be real (or at least genuinely labeled) sessions --
    this command does not generate data. See docs/EVALUATION_GUIDE.md for
    how to build one with mlcfq.evaluation.capture_harness and
    mlcfq.evaluation.adversarial.
    """
    from mlcfq.evaluation.dataset import load_labeled_sessions
    from mlcfq.evaluation.run_study import format_results_markdown, run_study

    dataset = load_labeled_sessions(dataset_path)
    profiles = load_profiles(profiles_path)
    results = run_study(dataset, profiles)
    report = format_results_markdown(results)

    if out:
        with open(out, "w") as f:
            f.write(report)
        click.echo(f"Wrote report -> {out}")
    else:
        click.echo(report)


if __name__ == "__main__":
    main()
