#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_FRAGMENTS = (
    "/simurgh",
    "/afs/",
    "/sailhome/",
    "/home/",
    "davidy02",
    "yuheng",
    "notes/",
    "deployment/scripts/",
    "openwam-data/libero-oxe-pretrain-5k",
    "examples/inference_libero_oxe.md",
    "Yao Feng",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage public MkDocs sources.")
    parser.add_argument("--output", default=".docs_site", help="Generated docs source directory.")
    args = parser.parse_args()

    source = REPO_ROOT / "docs"
    output = REPO_ROOT / args.output
    if not source.is_dir():
        raise SystemExit("docs/ is missing.")

    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(source, output)

    leaks: list[str] = []
    for path in output.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        matched = [fragment for fragment in FORBIDDEN_FRAGMENTS if fragment in text]
        if matched:
            rel = path.relative_to(output)
            leaks.append(f"{rel}: {', '.join(matched)}")

    if leaks:
        joined = "\n".join(leaks)
        raise SystemExit(f"Generated docs contain private path fragments:\n{joined}")

    try:
        display_path = output.relative_to(REPO_ROOT)
    except ValueError:
        display_path = output
    print(f"staged public docs in {display_path}")


if __name__ == "__main__":
    main()
