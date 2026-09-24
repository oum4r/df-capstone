"""Tests for the data path: load.resolve_data_dir and bundle_data.bundle."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analytics"))

import bundle_data  # noqa: E402
import load  # noqa: E402


def test_resolve_data_dir_env_wins(tmp_path):
    assert load.resolve_data_dir(str(tmp_path / "env"), tmp_path / "bundle") == tmp_path / "env"


def test_resolve_data_dir_falls_back_to_bundle(tmp_path):
    bundled = tmp_path / "bundle"
    assert load.resolve_data_dir(None, bundled) == bundled
    assert load.resolve_data_dir("", bundled) == bundled


def test_bundle_copies_files_and_slims_predictions(tmp_path):
    src = tmp_path / "src" / "processed"
    src.mkdir(parents=True)
    for name in bundle_data.COPIED:
        (src / name).write_bytes(b"x" + name.encode())
    wide = pd.DataFrame({"work_key": ["OL1W", "OL2W"], "y": [0.2, 0.4], "hist_gb": [0.25, 0.35], "catboost": [0.3, 0.3]})
    for name in bundle_data.SLIMMED:
        wide.to_parquet(src / name, index=False)

    rows = bundle_data.bundle(tmp_path / "src", tmp_path / "out")

    out = tmp_path / "out" / "processed"
    assert {r["file"] for r in rows} == set(bundle_data.COPIED) | set(bundle_data.SLIMMED)
    for name in bundle_data.COPIED:
        assert (out / name).read_bytes() == (src / name).read_bytes()
    for name in bundle_data.SLIMMED:
        slim = pd.read_parquet(out / name)
        assert list(slim.columns) == list(bundle_data.OOF_COLUMNS)
        assert slim["hist_gb"].tolist() == [0.25, 0.35]
    manifest = json.loads((tmp_path / "out" / "MANIFEST.json").read_text(encoding="utf-8"))
    assert len(manifest["files"]) == len(rows) and all(len(r["sha256"]) == 64 for r in manifest["files"])
