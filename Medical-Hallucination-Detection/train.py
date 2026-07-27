"""Training entry point scaffold.

This script initializes project infrastructure only. It loads configuration,
sets up logging, seeds random number generators, detects the compute device,
and prints the loaded configuration. Model training is intentionally not
implemented yet.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from pprint import pformat

from src.utils.config import Config, load_config
from src.utils.device import get_device
from src.utils.logging import setup_logging
from src.utils.seed import set_random_seed


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the infrastructure entry point."""
    parser = argparse.ArgumentParser(
        description="Initialize the medical hallucination detection pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def initialize_logging(config: Config) -> logging.Logger:
    """Initialize logging from configuration values.

    Args:
        config: Loaded project configuration.

    Returns:
        Configured root logger.
    """
    logging_config = config.get("logging", {})
    paths_config = config.get("paths", {})

    return setup_logging(
        log_dir=paths_config.get("logs_dir", "outputs/logs"),
        file_name=logging_config.get("file_name", "training.log"),
        level=logging_config.get("level", "INFO"),
        log_format=logging_config.get(
            "format", "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ),
        date_format=logging_config.get("date_format", "%Y-%m-%d %H:%M:%S"),
    )


def main() -> None:
    """Run the project initialization workflow without training a model."""
    args = parse_args()
    config = load_config(args.config)
    logger = initialize_logging(config)

    seed = int(config.get("seed", 42))
    set_random_seed(seed)

    device = get_device()

    logger.info("Loaded configuration from %s", args.config)
    logger.info("Random seed set to %s", seed)
    logger.info("Detected device: %s", device)
    logger.info("Loaded configuration:\n%s", pformat(config, sort_dicts=False))
    logger.info("Training is not implemented yet.")


if __name__ == "__main__":
    main()

