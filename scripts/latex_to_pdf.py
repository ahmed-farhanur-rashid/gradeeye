#!/usr/bin/env python3
"""Render LaTeX docs to PDF using a single pdflatex/lualatex pass + bibtex if .bib.

Tries `tectonic` first (no extra setup), falls back to `pdflatex`/`bibtex`/`pdflatex`/`pdflatex`.

Usage:
    python scripts/latex_to_pdf.py                       # build all .tex in docs/{analysis,methodology}
    python scripts/latex_to_pdf.py docs/analysis/analysis.tex
    python scripts/latex_to_pdf.py docs/methodology/methodology.tex
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRS = [
    REPO_ROOT / "docs/analysis",
    REPO_ROOT / "docs/methodology",
]


def find_tex_targets(args_targets: list[str]) -> list[Path]:
    if args_targets:
        return [Path(t).resolve() for t in args_targets]
    out = []
    for d in DEFAULT_DIRS:
        if d.is_dir():
            out.extend(sorted(d.glob("*.tex")))
    return out


def has_bib(tex: Path) -> bool:
    return any(tex.parent.glob("*.bib"))


def run_tectonic(tex: Path, out_dir: Path) -> bool:
    if not shutil.which("tectonic"):
        return False
    cmd = [
        "tectonic",
        "--keep-intermediates",
        "--keep-logs",
        "--outdir", str(out_dir),
        str(tex),
    ]
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("  TECTONIC FAILED:\n" + r.stdout[-2000:] + "\n" + r.stderr[-2000:])
        return False
    return True


def run_pdflatex(tex: Path, out_dir: Path, need_bib: bool) -> bool:
    if not shutil.which("pdflatex"):
        return False
    passes = ["pdflatex"] * (3 if need_bib else 2)
    if need_bib and shutil.which("bibtex"):
        cmd_seq = [["pdflatex"], ["bibtex"], ["pdflatex"], ["pdflatex"]]
    else:
        cmd_seq = [["pdflatex"], ["pdflatex"]]
    for tool in cmd_seq:
        cmd = tool + [
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory", str(out_dir),
            str(tex),
        ]
        if tool[0] == "bibtex":
            cmd = [tool[0], str(tex.with_suffix("").name)]
        print(f"  $ {' '.join(cmd)}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  {tool[0]} FAILED:\n" + r.stdout[-2000:] + "\n" + r.stderr[-2000:])
            return False
    return True


def build_one(tex: Path) -> bool:
    out_dir = tex.parent
    print(f"\nBuilding {tex.relative_to(REPO_ROOT)} ...")
    need_bib = has_bib(tex)
    # Try tectonic first
    if run_tectonic(tex, out_dir):
        pdf = tex.with_suffix(".pdf")
        if pdf.exists():
            print(f"  ✓ {pdf.relative_to(REPO_ROOT)} ({pdf.stat().st_size/1024:.1f} KB)")
            return True
    # Fall back to pdflatex
    if run_pdflatex(tex, out_dir, need_bib):
        pdf = tex.with_suffix(".pdf")
        if pdf.exists():
            print(f"  ✓ {pdf.relative_to(REPO_ROOT)} ({pdf.stat().st_size/1024:.1f} KB)")
            return True
    print(f"  ✗ FAILED to build {tex.name}. Install tectonic OR texlive-latex-recommended + texlive-latex-extra + texlive-fonts-recommended + texlive-publishers.")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", help="Specific .tex files (default: all under docs/{analysis,methodology})")
    args = ap.parse_args()
    targets = find_tex_targets(args.targets)
    if not targets:
        print("No .tex files found.")
        sys.exit(1)
    print(f"Will build {len(targets)} .tex file(s):")
    for t in targets:
        print(f"  - {t.relative_to(REPO_ROOT)}")
    ok = 0
    for t in targets:
        if build_one(t):
            ok += 1
    print(f"\nDone: {ok}/{len(targets)} built.")


if __name__ == "__main__":
    main()