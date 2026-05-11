from __future__ import annotations

from bridge_retrofit.cli import build_parser


def test_global_project_root_not_clobbered_by_subparser_defaults():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--config",
            "configs/colab_large.yaml",
            "--project-root",
            "/content/drive/MyDrive/retrofit",
            "preprocess",
        ]
    )

    assert args.config == "configs/colab_large.yaml"
    assert args.project_root == "/content/drive/MyDrive/retrofit"


def test_subcommand_allows_project_root_after_subcommand():
    parser = build_parser()

    args = parser.parse_args(
        [
            "preprocess",
            "--project-root",
            "/content/drive/MyDrive/retrofit",
        ]
    )

    assert args.command == "preprocess"
    assert args.project_root == "/content/drive/MyDrive/retrofit"
