#!/usr/bin/env python3
"""Vox orchestrator entry point."""

from apollo_home import apollo_path
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(apollo_path("logs/orchestrator.log")),
    ],
)

from pipeline import ConversationPipeline


def main():
    parser = argparse.ArgumentParser(description="Vox live voice conversationalist")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    Path(apollo_path("logs")).mkdir(parents=True, exist_ok=True)
    pipeline = ConversationPipeline(args.config)
    pipeline.run()


if __name__ == "__main__":
    main()
