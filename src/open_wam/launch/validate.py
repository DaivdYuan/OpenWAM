from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from open_wam.data.latent_factory import build_train_val_latent_datasets
from open_wam.training import load_training_cli_config, parse_train_cli

from .matrix import load_launch_matrix, select_launch_specs
from .planning import build_launch_jobs, preflight_launch_job, validate_launch_job
from .wrappers import resolve_wrapper_train_argv


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MATRIX = REPO_ROOT / "configs" / "launch" / "marlowe_m1_m5_cross_domain.yaml"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate an Open-WAM launch matrix before scheduler submission.")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--cluster", type=str, default="marlowe")
    parser.add_argument("--cases", type=str, default="all")
    parser.add_argument("--num-steps", type=int, default=4000)
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/openwam_launch_validate"))
    parser.add_argument("--account", type=str, default=None)
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=("offline", "online", "disabled"))
    parser.add_argument(
        "--skip-filesystem-preflight",
        action="store_true",
        help="Only validate matrix references and final ExperimentConfig loading.",
    )
    parser.add_argument(
        "--build-dataset-smoke",
        action="store_true",
        help="Also instantiate train/val latent datasets after config loading. This can touch real data.",
    )
    args = parser.parse_args(argv)

    matrix = load_launch_matrix(args.matrix)
    if args.cluster not in matrix.clusters:
        raise ValueError(f"Unknown cluster {args.cluster!r}; available: {', '.join(matrix.clusters)}")
    cluster = matrix.clusters[args.cluster]
    specs = select_launch_specs(matrix, args.cases)
    jobs = build_launch_jobs(
        matrix=matrix,
        cluster=cluster,
        specs=specs,
        output_root=args.output_root,
        num_steps=args.num_steps,
        account=args.account,
        partition=args.partition,
        wandb_project=args.wandb_project or matrix.default_wandb_project,
        wandb_mode=args.wandb_mode,
        wandb_entity=args.wandb_entity,
    )
    validations = [validate_launch_job(job=job, matrix=matrix, repo_root=REPO_ROOT) for job in jobs]
    preflight_rows = [] if args.skip_filesystem_preflight else [preflight_launch_job(job) for job in jobs]
    dataset_smoke_rows = [dataset_smoke(job) for job in jobs] if args.build_dataset_smoke else []
    errors = [error for validation in validations for error in validation.errors]
    if not args.skip_filesystem_preflight:
        for row in preflight_rows:
            errors.extend(f"{row['key']}: {item['kind']} missing at {item['path']}" for item in row["missing"])
    for row in dataset_smoke_rows:
        errors.extend(f"{row['key']}: dataset smoke failed: {error}" for error in row["errors"])
    payload: dict[str, Any] = {
        "status": "ok" if not errors else "failed",
        "validated_at": datetime.now().isoformat(),
        "matrix": str(args.matrix),
        "cluster": cluster.name,
        "cases": [job.spec.name for job in jobs],
        "validation": [
            {
                "key": validation.key,
                "config_name": validation.config_name,
                "method_profile": validation.method_profile,
                "policy_variant_type": validation.policy_variant_type,
                "ok": validation.ok,
                "errors": list(validation.errors),
            }
            for validation in validations
        ],
        "preflight": preflight_rows,
        "dataset_smoke": dataset_smoke_rows,
        "errors": errors,
    }
    print(json.dumps(payload, indent=2))
    if errors:
        raise SystemExit(1)


def dataset_smoke(job) -> dict[str, Any]:
    try:
        config = load_training_cli_config(
            parse_train_cli(resolve_wrapper_train_argv(job, repo_root=REPO_ROOT)),
            env=job.env,
        )
        train_dataset, val_dataset = build_train_val_latent_datasets(config.data)
        return {
            "key": job.spec.name,
            "train_len": len(train_dataset),
            "val_len": len(val_dataset),
            "errors": [],
        }
    except Exception as exc:
        return {
            "key": job.spec.name,
            "train_len": None,
            "val_len": None,
            "errors": [str(exc)],
        }


if __name__ == "__main__":
    main()
