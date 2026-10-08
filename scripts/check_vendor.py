"""Verify pinned source files and preserved permissive licenses without network."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
records = json.loads((root / "third_party/vendor-files.json").read_text())
expected = {r["path"] for r in records}
actual = {p.relative_to(root).as_posix() for p in (root / "vendor").rglob("*") if p.is_file()}
assert expected == actual, "Vendor file list changed; review provenance before updating manifest"
for record in records:
    assert hashlib.sha256((root / record["path"]).read_bytes()).hexdigest() == record["sha256"], record["path"]
for directory in ("vendor/qwentts", "vendor/ggml"):
    assert "Permission is hereby granted" in (root / directory / "LICENSE").read_text()
print(f"Verified {len(records)} pinned vendor source files and MIT notices")
