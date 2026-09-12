"""x_A: HTTP/3 SETTINGS / QPACK / stream behavior feature layer (paper §5.1)."""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

from mlcfq.model import Http3Features

# Canonical HTTP/3 SETTINGS identifiers (RFC 9114 §7.2.4, plus QPACK's
# RFC 9204) — used to normalize numeric setting IDs to readable names when
# extraction supplies raw codepoints.
SETTINGS_ID_NAMES = {
    0x01: "QPACK_MAX_TABLE_CAPACITY",
    0x06: "MAX_FIELD_SECTION_SIZE",
    0x07: "QPACK_BLOCKED_STREAMS",
    0x08: "ENABLE_CONNECT_PROTOCOL",
    0x33: "H3_DATAGRAM",
}


def build_http3_features(raw: Mapping[str, Any]) -> Http3Features:
    """Build x_A from a raw HTTP/3 frame/settings field dict."""
    data = dict(raw)

    settings = data.get("settings", {}) or {}
    normalized_settings: dict[str, int] = {}
    for key, value in settings.items():
        if isinstance(key, int):
            name = SETTINGS_ID_NAMES.get(key, f"0x{key:x}")
        else:
            name = str(key)
        try:
            normalized_settings[name] = int(value)
        except (TypeError, ValueError):
            continue
    data["settings"] = normalized_settings

    if "qpack_max_table_capacity" not in data:
        data["qpack_max_table_capacity"] = normalized_settings.get(
            "QPACK_MAX_TABLE_CAPACITY"
        )
    if "qpack_blocked_streams" not in data:
        data["qpack_blocked_streams"] = normalized_settings.get("QPACK_BLOCKED_STREAMS")

    known = {f.name for f in dataclasses.fields(Http3Features)}
    filtered = {k: v for k, v in data.items() if k in known}
    return Http3Features(**filtered)
