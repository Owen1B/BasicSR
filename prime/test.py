from __future__ import annotations

from pathlib import Path

from basicsr.test import test_pipeline

from prime import register_all


def main() -> None:
    register_all()
    root_path = Path(__file__).resolve().parents[1]
    test_pipeline(str(root_path))


if __name__ == "__main__":
    main()
