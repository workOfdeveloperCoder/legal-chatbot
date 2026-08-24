#!/usr/bin/env python3
"""Qdrant snapshot / verify helpers for production (no Docker)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def _base_url() -> str:
    url = (os.environ.get("QDRANT_URL") or "").strip()
    if url:
        return url.rstrip("/")
    host = os.environ.get("QDRANT_HOST", "127.0.0.1")
    port = os.environ.get("QDRANT_PORT", "6333")
    https = os.environ.get("QDRANT_HTTPS", "").lower() in {"1", "true", "yes"}
    scheme = "https" if https else "http"
    return f"{scheme}://{host}:{port}"


def snapshot_recover_location(
    collection: str,
    snapshot_ref: str,
    *,
    base_url: str | None = None,
) -> str:
    """
    Build a Qdrant recover `location` URI.

    Qdrant requires a URL or file:// URI, not a bare snapshot filename.
    See https://qdrant.tech/documentation/snapshots/
    """
    ref = (snapshot_ref or "").strip()
    if not ref:
        raise ValueError("snapshot location is required")
    if ref.startswith(("http://", "https://", "file://")):
        return ref
    if ref.startswith("/"):
        return f"file://{ref}"
    base = (base_url or _base_url()).rstrip("/")
    return f"{base}/collections/{collection}/snapshots/{ref}"


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = (os.environ.get("QDRANT_API_KEY") or "").strip()
    if key:
        headers["api-key"] = key
    return headers


def _request(method: str, path: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{_base_url()}{path}",
        data=data,
        headers=_headers(),
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Qdrant HTTP {exc.code} {path}: {detail}") from exc


def snapshot(collection: str) -> None:
    result = _request("POST", f"/collections/{collection}/snapshots")
    print(json.dumps(result, indent=2))


def list_snapshots(collection: str) -> None:
    result = _request("GET", f"/collections/{collection}/snapshots")
    print(json.dumps(result, indent=2))


def verify(
    collection: str,
    expected_size: int,
    *,
    check_payload: bool = False,
) -> None:
    result = _request("GET", f"/collections/{collection}")
    result_body = result.get("result") or result
    points = result_body.get("points_count")
    vectors = (
        ((result_body.get("config") or {}).get("params") or {}).get("vectors")
        or {}
    )
    size = vectors.get("size")
    distance = vectors.get("distance")
    report: dict[str, object] = {
        "collection": collection,
        "points_count": points,
        "vector_size": size,
        "distance": distance,
        "ok_size": size == expected_size,
        "ok_exists": points is not None,
    }
    if check_payload and points:
        scroll = _request(
            "POST",
            f"/collections/{collection}/points/scroll",
            {
                "limit": 1,
                "with_payload": True,
                "with_vector": False,
            },
        )
        scroll_body = scroll.get("result") or scroll
        batch = scroll_body.get("points") or []
        payload = (batch[0].get("payload") if batch else None) or {}
        report["sample_payload_keys"] = sorted(payload.keys())
        report["ok_payload"] = bool(payload)
    print(json.dumps(report, indent=2))
    if size != expected_size:
        raise SystemExit(
            f"Vector size mismatch: got {size}, expected {expected_size}"
        )
    if not points:
        raise SystemExit(f"Collection {collection} has no points")
    if check_payload and not report.get("ok_payload", True):
        raise SystemExit(f"Collection {collection} sample point has empty payload")


def recover(collection: str, snapshot_name: str) -> None:
    location = snapshot_recover_location(collection, snapshot_name)
    result = _request(
        "PUT",
        f"/collections/{collection}/snapshots/recover",
        {
            "location": location,
            "priority": "snapshot",
        },
    )
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Qdrant production ops")
    sub = parser.add_subparsers(dest="cmd", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("collection")
    listed = sub.add_parser("list-snapshots")
    listed.add_argument("collection")
    ver = sub.add_parser("verify")
    ver.add_argument("collection")
    ver.add_argument("--vector-size", type=int, default=768)
    ver.add_argument(
        "--payload",
        action="store_true",
        help="Scroll one point and report payload keys",
    )
    rec = sub.add_parser("recover")
    rec.add_argument("collection")
    rec.add_argument(
        "snapshot_name",
        help=(
            "Snapshot filename on this Qdrant node, file:///absolute/path.snapshot, "
            "or http(s) URL"
        ),
    )
    args = parser.parse_args()
    if args.cmd == "snapshot":
        snapshot(args.collection)
    elif args.cmd == "list-snapshots":
        list_snapshots(args.collection)
    elif args.cmd == "verify":
        verify(args.collection, args.vector_size, check_payload=args.payload)
    elif args.cmd == "recover":
        recover(args.collection, args.snapshot_name)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
