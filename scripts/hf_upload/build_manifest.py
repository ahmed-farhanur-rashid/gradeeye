"""Build the per-repo upload manifest for the staged Hugging Face upload.

For every staged file: SHA-256, byte size, target repo, relative path.
Writes:
  - staging/hf/<repo>/UPLOAD_MANIFEST.csv (per repo)
  - staging/hf/_all_manifest.csv (combined)
  - staging/hf/_manifest_summary.json (per-repo counts)

This stage does NOT contact Hugging Face.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STAGING = REPO / "staging/hf"

REPO_TARGETS = {
    "baseline-3ch": "gradeeye/baseline-3ch",
    "baseline-3ch-unbalanced": "gradeeye/baseline-3ch-unbalanced",
    "four-ch-morph": "gradeeye/four-ch-morph",
    "four-ch-soft": "gradeeye/four-ch-soft",
    "four-ch-tversky": "gradeeye/four-ch-tversky",
    "seg-unet-bcedice": "gradeeye/seg-unet-bcedice",
    "seg-unet-tversky": "gradeeye/seg-unet-tversky",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", default=str(STAGING))
    args = ap.parse_args()
    staging = Path(args.staging)
    if not staging.is_dir():
        raise SystemExit(f"missing {staging}; run convert_to_safetensors.py first")

    # Per-repo tables.
    summary: dict[str, dict] = {}
    for repo_dir in sorted(p for p in staging.iterdir() if p.is_dir()):
        if repo_dir.name not in REPO_TARGETS:
            continue
        repo = repo_dir.name
        repo_id = REPO_TARGETS[repo]
        rows = []
        for f in sorted(repo_dir.rglob("*")):
            if not f.is_file():
                continue
            if f.name == "UPLOAD_MANIFEST.csv":
                continue
            rel = f.relative_to(repo_dir).as_posix()
            rows.append({
                "repo_id": repo_id,
                "path": rel,
                "bytes": f.stat().st_size,
                "sha256": sha256(f),
            })
        # Write per-repo CSV.
        csv_path = repo_dir / "UPLOAD_MANIFEST.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["repo_id", "path", "bytes", "sha256"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        summary[repo] = {
            "repo_id": repo_id,
            "n_files": len(rows),
            "total_bytes": sum(r["bytes"] for r in rows),
            "manifest_csv": str(csv_path.relative_to(REPO)),
        }

    # Combined manifest.
    all_rows = []
    for repo_dir in sorted(p for p in staging.iterdir() if p.is_dir()):
        if repo_dir.name not in REPO_TARGETS:
            continue
        with (repo_dir / "UPLOAD_MANIFEST.csv").open() as f:
            for row in csv.DictReader(f):
                all_rows.append(row)
    all_csv = staging / "_all_manifest.csv"
    with all_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["repo_id", "path", "bytes", "sha256"])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    summary_path = staging / "_manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Total files to upload: {len(all_rows)}")
    print(f"Total bytes:           {sum(int(r['bytes']) for r in all_rows):,}")
    print()
    for repo, s in summary.items():
        mb = s["total_bytes"] / (1024 * 1024)
        print(f"  {s['repo_id']:<40} {s['n_files']:>3} files  {mb:>8.1f} MB")
    print()
    print(f"Wrote {all_csv.relative_to(REPO)} and {summary_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
