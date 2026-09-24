"""Copy the data the dashboard reads into a folder that can ship with it.

Writes every file the pages read to <out>/processed and a MANIFEST.json with
each file's size and SHA-256. The two out-of-fold prediction files are cut to
the three columns the scatter reads; every other file is copied byte for byte.

    python analytics/bundle_data.py --source <data root> --out analytics/data

--source defaults to CAPSTONE_DATA.
"""

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

COPIED = (
    "blend_weight_day7.json",
    "explain_day5.json",
    "features.json",
    "learning_curve.json",
    "llm_eval.json",
    "metrics_day3.json",
    "metrics_day4.json",
    "metrics_day5.json",
    "profile_features_day7.json",
    "r2_investigation.json",
    "sanity_checks_day6.json",
    "shelf_by_month.parquet",
    "training.parquet",
)
OOF_COLUMNS = ("work_key", "y", "hist_gb")
SLIMMED = {f"sweep_day5_{target}_oof.parquet": OOF_COLUMNS for target in ("share", "ratio")}


def sha256(path: Path) -> str:
    """SHA-256 of a file, hex.

    :param path: the file.
    :return: the digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bundle(source: Path, out: Path) -> list[dict]:
    """Copy and slim the dashboard's files from source/processed to out/processed.

    :param source: the data root holding processed/.
    :param out: the bundle root; processed/ and MANIFEST.json are written here.
    :return: one manifest row per file written.
    """
    src, dst = source / "processed", out / "processed"
    dst.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in COPIED:
        shutil.copyfile(src / name, dst / name)
        rows.append({"file": name, "how": "copied"})
    for name, columns in SLIMMED.items():
        pd.read_parquet(src / name, columns=list(columns)).to_parquet(dst / name, index=False)
        rows.append({"file": name, "how": f"columns {', '.join(columns)}"})
    for row in rows:
        path = dst / row["file"]
        row.update({"bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {"built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": rows}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return rows


def main() -> None:
    """Command line: bundle from --source (or CAPSTONE_DATA) into --out."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=os.getenv("CAPSTONE_DATA"), help="data root holding processed/")
    parser.add_argument("--out", type=Path, required=True, help="bundle root, e.g. analytics/data")
    args = parser.parse_args()
    if args.source is None:
        parser.error("give --source or set CAPSTONE_DATA")
    rows = bundle(Path(args.source), args.out)
    total = sum(r["bytes"] for r in rows)
    print(f"{len(rows)} files, {total / 1_048_576:.1f} MB, written to {args.out}")


if __name__ == "__main__":
    main()
