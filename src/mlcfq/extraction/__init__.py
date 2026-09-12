"""Passive capture extraction: pcap → structured Session records (paper §6).

The live extractor (`pcap_extractor.py`) requires `tshark`/`pyshark` and is
optional — the rest of the pipeline (features, scoring, cohort) operates on
`mlcfq.model.Session` objects regardless of how they were produced, so tests
and examples use hand-authored JSON fixtures instead of live capture.
"""

from mlcfq.extraction.pcap_extractor import PcapExtractionError, extract_sessions_from_pcap

__all__ = ["extract_sessions_from_pcap", "PcapExtractionError"]
