from __future__ import annotations

from pathlib import Path

from basicsr.train import train_pipeline

from prime import register_all


def main() -> None:
    register_all()
    root_path = Path(__file__).resolve().parents[1]
    train_pipeline(str(root_path))


if __name__ == "__main__":
    main()
