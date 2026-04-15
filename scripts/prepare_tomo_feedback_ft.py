from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


VIEWS = 60
HEIGHT = 128
WIDTH = 128


@dataclass(frozen=True)
class FeedbackCase:
    case_id: str
    slug: str
    rel_path: str


FEEDBACK_CASES = (
    FeedbackCase("BONE1_tomo", "bone1tomo", "BONE1/tomo/projection_raw.dat"),
    FeedbackCase("BONE2_tomo_1", "bone2tomo1", "BONE2/tomo-1/projection_raw.dat"),
    FeedbackCase("BONE2_tomo_2", "bone2tomo2", "BONE2/tomo-2/projection_raw.dat"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare staged tomo feedback finetune datasets and manifests."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="BasicSR repo root. Defaults to the current script's repo root.",
    )
    parser.add_argument(
        "--spect229-root",
        type=Path,
        default=None,
        help="Root containing SPECT229 patient folders with *_Proj4Filter.dat files.",
    )
    parser.add_argument(
        "--feedback-root",
        type=Path,
        default=None,
        help="Root containing feedback_cases/BONE*/tomo*/projection_raw.dat files.",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        default=None,
        help="Output root for staged tomo finetune training files.",
    )
    parser.add_argument(
        "--eval-out",
        type=Path,
        default=None,
        help="Output root for staged tomo finetune eval files.",
    )
    parser.add_argument(
        "--analysis-out",
        type=Path,
        default=None,
        help="Output root for manifests and selection summaries.",
    )
    parser.add_argument(
        "--train-start",
        type=int,
        default=4,
        help="Inclusive start index into sorted SPECT229 files for the train-anchor pool.",
    )
    parser.add_argument(
        "--train-end",
        type=int,
        default=229,
        help="Exclusive end index into sorted SPECT229 files for the train-anchor pool.",
    )
    parser.add_argument(
        "--anchor-bins",
        type=int,
        default=8,
        help="Number of count bins used to sample train anchors.",
    )
    parser.add_argument(
        "--anchors-per-bin",
        type=int,
        default=2,
        help="How many train anchors to take from each count bin.",
    )
    parser.add_argument(
        "--eval-anchor-count",
        type=int,
        default=4,
        help="How many held-out eval anchors to sample from the remaining pool.",
    )
    parser.add_argument(
        "--feedback-repeat",
        type=int,
        default=4,
        help="How many train copies to stage for each feedback tomo case.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How to materialize staged files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove any existing staged train/eval/analysis roots before preparing new outputs.",
    )
    return parser.parse_args()


def resolve_default_paths(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    if args.spect229_root is None:
        args.spect229_root = repo_root / "datasets" / "SPECT229"
    if args.feedback_root is None:
        args.feedback_root = repo_root / "datasets" / "feedback_cases"
    if args.train_out is None:
        args.train_out = repo_root / "datasets" / "feedback_tomo_ft_train"
    if args.eval_out is None:
        args.eval_out = repo_root / "datasets" / "feedback_tomo_ft_eval"
    if args.analysis_out is None:
        args.analysis_out = repo_root / "analysis" / "feedback_ft"


def load_projection_counts(path: Path) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.uint16)
    expected = VIEWS * HEIGHT * WIDTH
    if arr.size != expected:
        raise ValueError(f"Invalid tomo projection size for {path}: got {arr.size}, expected {expected}")
    return arr.reshape(VIEWS, HEIGHT, WIDTH)


def projection_stats(path: Path) -> dict[str, float | int]:
    proj = load_projection_counts(path)
    return {
        "sum": float(proj.sum(dtype=np.float64)),
        "max": int(proj.max(initial=0)),
        "mean": float(proj.mean(dtype=np.float64)),
        "nonzero": int(np.count_nonzero(proj)),
    }


def list_spect229_projection_files(root: Path) -> list[Path]:
    files = sorted(p for p in root.glob("**/*_Proj4Filter.dat") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No *_Proj4Filter.dat files found under {root}")
    return files


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def materialize_file(src: Path, dst: Path, link_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link_mode == "symlink":
        os.symlink(src.resolve(), dst)
        return
    if link_mode == "hardlink":
        os.link(src, dst)
        return
    shutil.copy2(src, dst)


def quantile_positions(count: int, picks: int) -> list[int]:
    if count <= 0 or picks <= 0:
        return []
    if count == 1:
        return [0]
    values = np.linspace(0.2, 0.8, num=picks)
    return sorted({min(count - 1, max(0, int(round(v * (count - 1))))) for v in values})


def select_anchor_records(
    records: list[dict[str, object]],
    *,
    anchor_bins: int,
    anchors_per_bin: int,
    eval_anchor_count: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ordered = sorted(records, key=lambda item: float(item["stats"]["sum"]))  # type: ignore[index]
    index_chunks = [chunk.tolist() for chunk in np.array_split(np.arange(len(ordered)), anchor_bins) if len(chunk) > 0]

    train_indices: list[int] = []
    for chunk in index_chunks:
        for pos in quantile_positions(len(chunk), anchors_per_bin):
            idx = chunk[pos]
            if idx not in train_indices:
                train_indices.append(idx)

    used = set(train_indices)
    remaining = [idx for idx in range(len(ordered)) if idx not in used]

    eval_indices: list[int] = []
    for pos in quantile_positions(len(remaining), eval_anchor_count):
        idx = remaining[pos]
        if idx not in eval_indices:
            eval_indices.append(idx)

    train_records = [ordered[idx] for idx in train_indices]
    eval_records = [ordered[idx] for idx in eval_indices]
    return train_records, eval_records


def relative_to(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path.resolve())


def stage_feedback_train_case(
    case: FeedbackCase,
    src: Path,
    train_out: Path,
    repeat_count: int,
    link_mode: str,
    repo_root: Path,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    stats = projection_stats(src)
    for rep_idx in range(repeat_count):
        slug = f"{case.slug}r{rep_idx:02d}"
        dst = train_out / "train_fail" / slug / f"{slug}_Proj4Filter.dat"
        materialize_file(src, dst, link_mode)
        entries.append(
            {
                "case_id": case.case_id,
                "slug": slug,
                "split": "train_fail",
                "source_path": str(src.resolve()),
                "staged_path": relative_to(repo_root, dst),
                "repeat_count": repeat_count,
                "stats": stats,
            }
        )
    return entries


def stage_feedback_eval_case(
    case: FeedbackCase,
    src: Path,
    eval_out: Path,
    link_mode: str,
    repo_root: Path,
) -> dict[str, object]:
    stats = projection_stats(src)
    dst = eval_out / "eval_fail" / case.slug / f"{case.slug}_Proj4Filter.dat"
    materialize_file(src, dst, link_mode)
    return {
        "case_id": case.case_id,
        "slug": case.slug,
        "split": "eval_fail",
        "source_path": str(src.resolve()),
        "staged_path": relative_to(repo_root, dst),
        "repeat_count": 1,
        "stats": stats,
    }


def stage_anchor_records(
    records: Iterable[dict[str, object]],
    *,
    split: str,
    out_root: Path,
    link_mode: str,
    repo_root: Path,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for record in records:
        patient = str(record["patient"])
        src = Path(str(record["path"]))
        dst = out_root / split / patient / f"{patient}_Proj4Filter.dat"
        materialize_file(src, dst, link_mode)
        entries.append(
            {
                "case_id": patient,
                "slug": patient,
                "split": split,
                "source_path": str(src.resolve()),
                "staged_path": relative_to(repo_root, dst),
                "repeat_count": 1,
                "stats": record["stats"],
            }
        )
    return entries


def main() -> None:
    args = parse_args()
    resolve_default_paths(args)

    repo_root = args.repo_root.resolve()
    spect229_root = args.spect229_root.resolve()
    feedback_root = args.feedback_root.resolve()
    train_out = args.train_out.resolve()
    eval_out = args.eval_out.resolve()
    analysis_out = args.analysis_out.resolve()

    if not spect229_root.exists():
        raise FileNotFoundError(f"SPECT229 root not found: {spect229_root}")
    if not feedback_root.exists():
        raise FileNotFoundError(f"Feedback root not found: {feedback_root}")

    for path in (train_out, eval_out, analysis_out):
        ensure_clean_dir(path, overwrite=args.overwrite)

    spect229_files = list_spect229_projection_files(spect229_root)
    train_pool = spect229_files[args.train_start : args.train_end]
    if not train_pool:
        raise ValueError("Empty train anchor pool after train_start/train_end slicing.")

    anchor_records: list[dict[str, object]] = []
    for path in train_pool:
        anchor_records.append(
            {
                "patient": path.stem.split("_", 1)[0],
                "path": str(path.resolve()),
                "stats": projection_stats(path),
            }
        )

    train_anchor_records, eval_anchor_records = select_anchor_records(
        anchor_records,
        anchor_bins=args.anchor_bins,
        anchors_per_bin=args.anchors_per_bin,
        eval_anchor_count=args.eval_anchor_count,
    )

    train_manifest: list[dict[str, object]] = []
    eval_manifest: list[dict[str, object]] = []

    for case in FEEDBACK_CASES:
        src = feedback_root / case.rel_path
        if not src.exists():
            raise FileNotFoundError(f"Missing feedback case source: {src}")
        train_manifest.extend(
            stage_feedback_train_case(
                case=case,
                src=src,
                train_out=train_out,
                repeat_count=args.feedback_repeat,
                link_mode=args.link_mode,
                repo_root=repo_root,
            )
        )
        eval_manifest.append(
            stage_feedback_eval_case(
                case=case,
                src=src,
                eval_out=eval_out,
                link_mode=args.link_mode,
                repo_root=repo_root,
            )
        )

    train_manifest.extend(
        stage_anchor_records(
            train_anchor_records,
            split="train_anchor",
            out_root=train_out,
            link_mode=args.link_mode,
            repo_root=repo_root,
        )
    )
    eval_manifest.extend(
        stage_anchor_records(
            eval_anchor_records,
            split="eval_anchor",
            out_root=eval_out,
            link_mode=args.link_mode,
            repo_root=repo_root,
        )
    )

    analysis_out.mkdir(parents=True, exist_ok=True)
    train_manifest_path = analysis_out / "tomo_train_manifest.json"
    eval_manifest_path = analysis_out / "tomo_eval_manifest.json"
    summary_path = analysis_out / "tomo_feedback_selection_summary.json"

    train_manifest_path.write_text(json.dumps(train_manifest, indent=2), encoding="utf-8")
    eval_manifest_path.write_text(json.dumps(eval_manifest, indent=2), encoding="utf-8")

    summary = {
        "repo_root": str(repo_root),
        "spect229_root": str(spect229_root),
        "feedback_root": str(feedback_root),
        "train_out": str(train_out),
        "eval_out": str(eval_out),
        "analysis_out": str(analysis_out),
        "train_pool_count": len(train_pool),
        "train_manifest_count": len(train_manifest),
        "eval_manifest_count": len(eval_manifest),
        "feedback_repeat": args.feedback_repeat,
        "anchor_bins": args.anchor_bins,
        "anchors_per_bin": args.anchors_per_bin,
        "eval_anchor_count": args.eval_anchor_count,
        "train_anchor_patients": [str(item["patient"]) for item in train_anchor_records],
        "eval_anchor_patients": [str(item["patient"]) for item in eval_anchor_records],
        "feedback_cases": [asdict(case) for case in FEEDBACK_CASES],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Prepared tomo feedback finetune staging under {train_out} and {eval_out}")
    print(f"Train manifest: {train_manifest_path}")
    print(f"Eval manifest:  {eval_manifest_path}")
    print(f"Summary:        {summary_path}")


if __name__ == "__main__":
    main()
