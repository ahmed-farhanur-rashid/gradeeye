"""Upload each staged repo to its corresponding Hugging Face Hub repo.

Aborts on any unexpected repo state (non-empty when expected empty) unless
--allow-nonempty is set. Reports commit URLs per repo on success.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", default=str(STAGING))
    ap.add_argument("--allow-nonempty", action="store_true",
                    help="Allow upload to a non-empty repo. Default: abort.")
    ap.add_argument("--repo", action="append",
                    help="Restrict upload to a subset of repos (repeatable).")
    ap.add_argument("--summary-json", default="staging/hf/_upload_summary.json")
    args = ap.parse_args()
    staging = Path(args.staging)

    api = HfApi()
    user = api.whoami()
    print(f"Authenticated as: {user['name']} (org: gradeeye)\n")

    targets = REPO_TARGETS.copy()
    if args.repo:
        targets = {k: v for k, v in targets.items() if k in args.repo}
    if not targets:
        raise SystemExit("No repos selected")

    summary: dict[str, dict] = {}
    for repo_dir_name, repo_id in sorted(targets.items()):
        folder = staging / repo_dir_name
        if not folder.is_dir():
            print(f"SKIP {repo_id}: missing staging folder {folder}", file=sys.stderr)
            summary[repo_id] = {"status": "skipped_missing_staging"}
            continue

        # Pre-flight: check repo state.
        try:
            info = api.repo_info(repo_id, repo_type="model")
            siblings = list(info.siblings or [])
            if siblings and not args.allow_nonempty:
                print(f"ABORT {repo_id}: repo is non-empty "
                      f"({len(siblings)} files). Pass --allow-nonempty to override.",
                      file=sys.stderr)
                summary[repo_id] = {
                    "status": "aborted_nonempty",
                    "existing_files": len(siblings),
                }
                continue
        except Exception as e:
            print(f"ABORT {repo_id}: repo_info failed: {e}", file=sys.stderr)
            summary[repo_id] = {"status": "aborted_repo_info", "error": str(e)}
            continue

        # Sanity: count files in folder.
        file_count = sum(1 for p in folder.rglob("*") if p.is_file()
                         and p.name != "UPLOAD_MANIFEST.csv")
        if file_count == 0:
            print(f"ABORT {repo_id}: no files to upload", file=sys.stderr)
            summary[repo_id] = {"status": "aborted_empty"}
            continue

        # Upload.
        msg = f"Upload GradeEye {repo_dir_name} checkpoints"
        print(f"Uploading {repo_id} ({file_count} files from {folder.relative_to(REPO)})...")
        t0 = time.time()
        try:
            commit_info = api.upload_folder(
                folder_path=str(folder),
                repo_id=repo_id,
                repo_type="model",
                commit_message=msg,
                ignore_patterns=["UPLOAD_MANIFEST.csv", "_upload_summary.json"],
            )
            elapsed = time.time() - t0
            url = getattr(commit_info, "url", None) or str(commit_info)
            print(f"  OK in {elapsed:.1f}s: {url}")
            summary[repo_id] = {
                "status": "uploaded",
                "url": url,
                "files": file_count,
                "elapsed_s": round(elapsed, 1),
            }
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            summary[repo_id] = {"status": "failed", "error": str(e)}

    summary_path = REPO / args.summary_json
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n" + "=" * 80)
    print("UPLOAD SUMMARY")
    print("=" * 80)
    for repo_id, s in summary.items():
        status = s["status"]
        if "url" in s:
            print(f"  {repo_id:<40} {status:<10} {s['url']}")
        else:
            print(f"  {repo_id:<40} {status:<10} {s.get('error', '')}")
    print(f"\nWrote {summary_path.relative_to(REPO)}")

    # Exit non-zero if any repo failed or was aborted.
    if any(s["status"] not in ("uploaded",) for s in summary.values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())