"""Command-line interface.

Kept intentionally small; heavy lifting lives in library modules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bridge_retrofit.config import load_config


def _add_common_args(parser: argparse.ArgumentParser, *, suppress_defaults: bool) -> None:
    # Important: when we add the same options to both the main parser and subparsers,
    # argparse would otherwise apply subparser defaults *after* parsing the global args,
    # clobbering values like --project-root passed before the subcommand.
    default = argparse.SUPPRESS if suppress_defaults else None
    parser.add_argument(
        "--config",
        default=(argparse.SUPPRESS if suppress_defaults else "configs/default.yaml"),
        help="Path to YAML config (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--project-root",
        default=default,
        help="Override project_root from config (useful in Colab Drive).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bridge-retrofit")

    # Global flags.
    _add_common_args(parser, suppress_defaults=False)

    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("preprocess", help="Fit preprocessors and save processed dataset")
    _add_common_args(p_pre, suppress_defaults=True)

    p_train = sub.add_parser("train", help="Train a model")
    _add_common_args(p_train, suppress_defaults=True)
    p_train.add_argument("--task", choices=["severity", "retrofit"], required=True)

    p_sim = sub.add_parser("fit-similarity", help="Fit KNN similarity index")
    _add_common_args(p_sim, suppress_defaults=True)

    p_eval = sub.add_parser("evaluate", help="Evaluate pipeline components")
    _add_common_args(p_eval, suppress_defaults=True)

    p_pred = sub.add_parser("predict", help="Run single-record inference")
    _add_common_args(p_pred, suppress_defaults=True)
    p_pred.add_argument("--json", required=True, help="JSON object with feature values")

    p_serve = sub.add_parser("serve", help="Launch Gradio app (if installed)")
    _add_common_args(p_serve, suppress_defaults=True)
    p_serve.add_argument("--share", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(Path(args.config), project_root_override=args.project_root)

    # Lazy imports keep CLI responsive and allow partial installs.
    if args.command == "preprocess":
        from bridge_retrofit.pipeline import run_preprocess

        run_preprocess(cfg)
        return 0

    if args.command == "train":
        from bridge_retrofit.pipeline import run_train

        run_train(cfg, task=args.task)
        return 0

    if args.command == "fit-similarity":
        from bridge_retrofit.pipeline import run_fit_similarity

        run_fit_similarity(cfg)
        return 0

    if args.command == "evaluate":
        from bridge_retrofit.pipeline import run_evaluate

        run_evaluate(cfg)
        return 0

    if args.command == "predict":
        from bridge_retrofit.pipeline import run_predict

        payload = json.loads(args.json)
        result = run_predict(cfg, payload)
        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.command == "serve":
        from bridge_retrofit.app.gradio_app import launch

        launch(cfg, share=bool(args.share))
        return 0

    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
