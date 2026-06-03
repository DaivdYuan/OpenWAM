#!/usr/bin/env python3
"""Download an Open-WAM checkpoint snapshot from Hugging Face.

Usage:
    # Inference weights only (~10 GB)
    python scripts/download_checkpoint.py --repo-id <org-or-user>/<repo> --mode inference

    # Full model bundle (~30 GB; no optimizer/scheduler training state)
    python scripts/download_checkpoint.py --repo-id <org-or-user>/<repo> --mode full

Set HF_TOKEN when downloading from a gated or private repository. This public
snapshot does not configure a default checkpoint repository.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_ROOT = Path("checkpoints")
REPO_ID_ENV = "OPEN_WAM_CHECKPOINT_REPO_ID"

# WHY two modes: inference only needs the exported safetensors (~10 GB),
# but fine-tuning needs model_state.pt (~20 GB) + config/metadata. The
# optimizer/scheduler training state is not part of the released HF bundle.
ALLOW_PATTERNS_BY_MODE = {
    "inference": [
        "transformer/*",
        "resolved_config.yaml",
        "README.md",
    ],
    "full": None,  # WHY None: download every released file in the repo
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--repo-id",
        default=os.environ.get(REPO_ID_ENV),
        help=f"Hugging Face repo id to download, or set {REPO_ID_ENV}.",
    )
    parser.add_argument(
        "--mode",
        choices=list(ALLOW_PATTERNS_BY_MODE),
        default="inference",
        help="Download mode: inference for safetensors only (~10 GB), full for the released model bundle (~30 GB).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Local directory for the downloaded checkpoint. Defaults to checkpoints/<repo-name>.",
    )
    args = parser.parse_args()

    if not args.repo_id:
        print(
            f"ERROR: no checkpoint repo configured. Pass --repo-id or set {REPO_ID_ENV}.",
            file=sys.stderr,
        )
        sys.exit(2)

    output_dir = args.output_dir or (DEFAULT_ROOT / args.repo_id.rsplit("/", 1)[-1])
    token = os.environ.get("HF_TOKEN")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    allow_patterns = ALLOW_PATTERNS_BY_MODE[args.mode]
    print(f"Downloading {args.repo_id} (mode={args.mode}) to {output_dir} ...")

    try:
        local_path = snapshot_download(
            repo_id=args.repo_id,
            local_dir=str(output_dir),
            allow_patterns=allow_patterns,
            token=token or None,
        )
    except Exception as exc:
        # WHY catch broadly: huggingface_hub raises different exceptions for
        # 401 (bad token), 403 (no access), 404 (repo not found). A clear
        # message helps users distinguish auth vs access issues.
        msg = str(exc)
        if "401" in msg or "403" in msg:
            print(
                f"ERROR: Access denied to {args.repo_id}.\n"
                "If this is a gated or private repo, set a valid HF_TOKEN with read access.",
                file=sys.stderr,
            )
        else:
            print(f"ERROR: Download failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Downloaded to: {local_path}")
    print("Done.")


if __name__ == "__main__":
    main()
