from pathlib import Path
import sys


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> None:
    _add_src_to_path()
    from phishing_hybrid.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
