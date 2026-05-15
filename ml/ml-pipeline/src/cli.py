"""Command-line entrypoints for the ML pipeline."""

from __future__ import annotations

import argparse
import json

from src.config import CONFIG
from src.exports.export_frontend_json import export_frontend_json
from src.features.feature_builder import build_and_save_features
from src.ingestion.load_matches import ingest_matches
from src.models.evaluate_models import evaluate_models
from src.models.train import train_models
from src.simulation.worldcup import download_worldcup_fixture, simulate_worldcup
from src.utils.logging_utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="International football ML pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ["ingest", "build-features", "train", "evaluate", "export", "run-all"]:
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--sample", action="store_true", help="Use built-in sample data")
    sim = subparsers.add_parser("simulate-worldcup")
    sim.add_argument("--runs", type=int, default=1000)
    sim.add_argument("--download-fixture", action="store_true")
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    CONFIG.ensure_directories()

    if args.command == "ingest":
        ingest_matches(sample=args.sample)
    elif args.command == "build-features":
        build_and_save_features(sample=args.sample)
    elif args.command == "train":
        print(json.dumps(train_models(), indent=2))
    elif args.command == "evaluate":
        print(json.dumps(evaluate_models(), indent=2))
    elif args.command == "export":
        print(json.dumps(export_frontend_json(), indent=2))
    elif args.command == "simulate-worldcup":
        if args.download_fixture:
            download_worldcup_fixture()
        print(json.dumps(simulate_worldcup(runs=args.runs), indent=2))
    elif args.command == "run-all":
        ingest_matches(sample=args.sample)
        build_and_save_features(sample=args.sample)
        train_models()
        evaluate_models()
        simulate_worldcup(runs=1000)
        export_frontend_json()


if __name__ == "__main__":
    main()
