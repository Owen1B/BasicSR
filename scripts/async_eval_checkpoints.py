#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Asynchronous checkpoint evaluator for prime.")
    p.add_argument("--test-opt", required=True, type=str, help="Base test YAML (e.g. options/test/converged/*.yml).")
    p.add_argument("--models-dir", type=str, default=None, help="Checkpoint directory. Auto-infer if omitted.")
    p.add_argument("--pattern", type=str, default="net_g_*.pth", help="Checkpoint glob pattern.")
    p.add_argument("--poll-seconds", type=int, default=60, help="Polling interval in seconds.")
    p.add_argument("--python", type=str, default="python", help="Python executable to run prime.test.")
    p.add_argument("--once", action="store_true", help="Evaluate current checkpoints once and exit.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    p.add_argument("--include-latest", action="store_true", help="Include net_g_latest.pth.")
    p.add_argument("--include-best", action="store_true", help="Include net_g_best.pth.")
    p.add_argument("--state-file", type=str, default=None, help="State JSON path. Default: <models-dir>/.async_eval_state.json")
    p.add_argument("--param-key-g", type=str, default=None, help="Override path.param_key_g in generated test YAML.")
    p.add_argument("--every-n-iters", type=int, default=0, help="Only evaluate ckpts with iter %% N == 0. 0 means evaluate all.")
    p.add_argument(
        "--backlog-policy",
        type=str,
        default="latest",
        choices=["latest", "window", "all"],
        help="How to handle pending queue when training is faster than evaluation.",
    )
    p.add_argument("--max-pending", type=int, default=2, help="Used when --backlog-policy=window.")
    p.add_argument(
        "--step-skip-policy",
        type=str,
        default="mark_done",
        choices=["mark_done", "keep_pending"],
        help="How to handle checkpoints skipped by --every-n-iters.",
    )
    p.add_argument(
        "--drop-policy",
        type=str,
        default="mark_done",
        choices=["mark_done", "keep_pending"],
        help="How to handle checkpoints dropped by backlog policy.",
    )
    p.add_argument("--max-retries-per-ckpt", type=int, default=2, help="Maximum retries for one checkpoint on failure.")
    p.add_argument(
        "--retry-exhausted-policy",
        type=str,
        default="mark_done",
        choices=["mark_done", "keep_pending"],
        help="How to handle checkpoints that exceeded max retries.",
    )
    p.add_argument(
        "--lock-file",
        type=str,
        default=None,
        help="Single-flight lock file path. Default: <models-dir>/.async_eval.lock",
    )
    p.add_argument(
        "--mp4-enable",
        action="store_true",
        help="After each successful async validation, also render MP4 via scripts/eval_models_mp4.py.",
    )
    p.add_argument(
        "--mp4-patient",
        action="append",
        default=[],
        help="Patient id for MP4 generation. Repeatable; supports comma-separated values.",
    )
    p.add_argument(
        "--mp4-script",
        type=str,
        default="scripts/eval_models_mp4.py",
        help="MP4 script path, relative to repo root or absolute.",
    )
    p.add_argument(
        "--mp4-out-dir",
        type=str,
        default=None,
        help="Output directory for async MP4s. Default: results/async_eval_mp4/<exp_name>/",
    )
    p.add_argument("--mp4-spect229-dir", type=str, default="datasets/SPECT229")
    p.add_argument("--mp4-proj-shape", type=str, default="60,128,128")
    p.add_argument("--mp4-proj-dtype", type=str, default="uint16", choices=["int16", "uint16", "float32"])
    p.add_argument("--mp4-thin-factors", type=str, default="2,3,4,5")
    p.add_argument("--mp4-seed", type=int, default=123)
    p.add_argument("--mp4-fps", type=float, default=10.0)
    p.add_argument("--mp4-turns", type=int, default=1)
    p.add_argument("--mp4-crf", type=int, default=18)
    p.add_argument("--mp4-device", type=str, default="cuda")
    p.add_argument("--mp4-model-label", type=str, default="ema")
    p.add_argument("--mp4-with-bm3d", action="store_true")
    p.add_argument("--mp4-allow-on-the-fly-cache", action="store_true")
    p.add_argument("--mp4-no-require-cached", action="store_true")
    return p.parse_args()


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid YAML object: {path}")
    return obj


def _save_yaml(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=False)


def _numeric_iter(path: Path) -> int | None:
    m = re.search(r"net_g_(\d+)\.pth$", path.name)
    if m:
        return int(m.group(1))
    return None


def _iter_sort_key(path: Path) -> tuple[int, int]:
    n = _numeric_iter(path)
    if n is not None:
        return (0, n)
    if path.name == "net_g_latest.pth":
        return (1, 10**18 - 1)
    if path.name == "net_g_best.pth":
        return (1, 10**18)
    return (2, 10**18 + 1)


def _resolve_models_dir(root_dir: Path, cfg: dict, user_models_dir: str | None) -> Path:
    if user_models_dir:
        p = Path(user_models_dir)
        return p if p.is_absolute() else (root_dir / p)

    path_opt = cfg.get("path", {}) or {}
    pretrain = path_opt.get("pretrain_network_g", None)
    if pretrain:
        pretrain_p = Path(str(pretrain))
        if not pretrain_p.is_absolute():
            pretrain_p = root_dir / pretrain_p
        return pretrain_p.parent

    name = str(cfg.get("name", "unknown"))
    return root_dir / "experiments" / name / "models"


def _list_checkpoints(models_dir: Path, pattern: str, include_latest: bool, include_best: bool) -> list[Path]:
    ckpts = [p for p in models_dir.glob(pattern) if p.is_file()]
    out: list[Path] = []
    for p in ckpts:
        if not include_latest and p.name == "net_g_latest.pth":
            continue
        if not include_best and p.name == "net_g_best.pth":
            continue
        out.append(p)
    out.sort(key=_iter_sort_key)
    return out


def _default_state() -> dict[str, Any]:
    return {"done": [], "failed_counts": {}}


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _default_state()
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return _default_state()
        out = _default_state()
        if isinstance(obj.get("done"), list):
            out["done"] = obj["done"]
        if isinstance(obj.get("failed_counts"), dict):
            out["failed_counts"] = obj["failed_counts"]
        return out
    except Exception:
        return _default_state()


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _build_eval_yaml(base_cfg: dict, ckpt_path: Path, param_key_g: str | None) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("path", {})
    cfg["path"]["pretrain_network_g"] = str(ckpt_path)
    if param_key_g:
        cfg["path"]["param_key_g"] = str(param_key_g)

    base_name = str(cfg.get("name", "eval"))
    cfg["name"] = f"{base_name}_async_{ckpt_path.stem}"
    return cfg


def _run_eval(root_dir: Path, python_exe: str, tmp_opt: Path, dry_run: bool) -> int:
    cmd = [python_exe, "-m", "prime.test", "-opt", str(tmp_opt)]
    print(f"[async-eval] run: {' '.join(cmd)}")
    if dry_run:
        return 0
    ret = subprocess.run(cmd, cwd=str(root_dir), check=False)
    return int(ret.returncode)


def _normalize_patients(values: list[str]) -> list[str]:
    out: list[str] = []
    for raw in values:
        for part in str(raw).split(","):
            name = part.strip()
            if name:
                out.append(name)
    # de-dup keep order
    seen: set[str] = set()
    dedup: list[str] = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        dedup.append(p)
    return dedup


def _resolve_mp4_script(root_dir: Path, script_path: str) -> Path:
    p = Path(str(script_path))
    if p.is_absolute():
        return p
    return root_dir / p


def _resolve_experiment_dir(models_dir: Path) -> Path:
    models_dir = Path(models_dir)
    if models_dir.name == "models":
        return models_dir.parent
    return models_dir.parent


def _find_config_for_experiment(exp_dir: Path) -> Path:
    exp_dir = Path(exp_dir)
    cand = exp_dir / f"{exp_dir.name}.yml"
    if cand.is_file():
        return cand
    cand2 = exp_dir / f"{exp_dir.name}.yaml"
    if cand2.is_file():
        return cand2
    ys = sorted([p for p in exp_dir.iterdir() if p.is_file() and p.suffix.lower() in [".yml", ".yaml"]])
    if len(ys) == 1:
        return ys[0]
    if len(ys) > 1:
        for p in ys:
            if exp_dir.name in p.stem:
                return p
        return ys[0]
    raise FileNotFoundError(f"No .yml/.yaml found under experiment dir: {exp_dir}")


def _run_mp4_for_ckpt(
    *,
    root_dir: Path,
    python_exe: str,
    args: argparse.Namespace,
    models_dir: Path,
    ckpt: Path,
    dry_run: bool,
) -> int:
    if not bool(args.mp4_enable):
        return 0

    patients = _normalize_patients(list(args.mp4_patient or []))
    if not patients:
        raise ValueError("--mp4-enable requires at least one --mp4-patient.")

    mp4_script = _resolve_mp4_script(root_dir, str(args.mp4_script))
    if not mp4_script.exists():
        raise FileNotFoundError(f"mp4 script not found: {mp4_script}")

    exp_dir = _resolve_experiment_dir(models_dir)
    exp_name = exp_dir.name
    cfg = _find_config_for_experiment(exp_dir)

    out_dir = Path(str(args.mp4_out_dir)) if args.mp4_out_dir else (root_dir / "results" / "async_eval_mp4" / exp_name)
    if not out_dir.is_absolute():
        out_dir = root_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rc_all = 0
    for patient in patients:
        out_path = out_dir / f"{patient}_{ckpt.stem}_{str(args.mp4_model_label)}.mp4"
        model_spec = f"{str(args.mp4_model_label)}={str(ckpt.resolve())}|{str(cfg.resolve())}"
        cmd = [
            str(python_exe),
            str(mp4_script),
            "--patient",
            str(patient),
            "--out",
            str(out_path),
            "--spect229-dir",
            str(args.mp4_spect229_dir),
            "--proj-shape",
            str(args.mp4_proj_shape),
            "--proj-dtype",
            str(args.mp4_proj_dtype),
            "--model-ckpt",
            model_spec,
            "--seed",
            str(int(args.mp4_seed)),
            "--thin-factors",
            str(args.mp4_thin_factors),
            "--fps",
            str(float(args.mp4_fps)),
            "--turns",
            str(int(args.mp4_turns)),
            "--crf",
            str(int(args.mp4_crf)),
            "--device",
            str(args.mp4_device),
        ]
        if bool(args.mp4_with_bm3d):
            cmd.append("--with-bm3d")
        if bool(args.mp4_allow_on_the_fly_cache):
            cmd.append("--allow-on-the-fly-cache")
        if bool(args.mp4_no_require_cached):
            cmd.append("--no-require-cached")

        print(f"[async-eval][mp4] run: {' '.join(cmd)}")
        if bool(dry_run):
            continue
        ret = subprocess.run(cmd, cwd=str(root_dir), check=False)
        if int(ret.returncode) != 0:
            rc_all = int(ret.returncode)
            print(f"[async-eval][mp4] failed({ret.returncode}): patient={patient} ckpt={ckpt.name}")
        else:
            print(f"[async-eval][mp4] done: {out_path}")
    return int(rc_all)


def _acquire_single_flight_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    fh.seek(0)
    fh.truncate(0)
    fh.write(str(os.getpid()))
    fh.flush()
    return fh


def _apply_every_n_filter(ckpts: list[Path], every_n: int) -> tuple[list[Path], list[Path]]:
    if every_n <= 0:
        return ckpts, []
    selected: list[Path] = []
    skipped: list[Path] = []
    for p in ckpts:
        n = _numeric_iter(p)
        if n is None:
            selected.append(p)
            continue
        if n % every_n == 0:
            selected.append(p)
        else:
            skipped.append(p)
    return selected, skipped


def _apply_backlog_policy(ckpts: list[Path], policy: str, max_pending: int) -> tuple[list[Path], list[Path]]:
    if not ckpts:
        return [], []
    ordered = sorted(ckpts, key=_iter_sort_key)
    if policy == "all":
        return ordered, []
    if policy == "latest":
        return [ordered[-1]], ordered[:-1]
    keep = max(1, int(max_pending))
    if keep >= len(ordered):
        return ordered, []
    return ordered[-keep:], ordered[:-keep]


def _paths_to_keys(paths: list[Path]) -> list[str]:
    return [_ckpt_key(p) for p in paths]


def _ckpt_key(path: Path) -> str:
    """Stable key for de-dup/retry state.

    - Iter checkpoints (net_g_<iter>.pth): use absolute path key.
    - Moving targets (latest/best): include file stat signature so updates retrigger evaluation.
    """
    p = Path(path).resolve()
    n = _numeric_iter(p)
    if n is not None:
        return str(p)
    try:
        st = p.stat()
        return f"{p}::mtime_ns={int(st.st_mtime_ns)}::size={int(st.st_size)}"
    except Exception:
        return str(p)


def _mark_done(done: set[str], paths: list[Path]) -> int:
    cnt = 0
    for key in _paths_to_keys(paths):
        if key in done:
            continue
        done.add(key)
        cnt += 1
    return cnt


def main() -> int:
    args = _parse_args()
    if args.every_n_iters < 0:
        raise ValueError("--every-n-iters must be >= 0")
    if args.max_pending <= 0:
        raise ValueError("--max-pending must be > 0")
    if args.max_retries_per_ckpt < 0:
        raise ValueError("--max-retries-per-ckpt must be >= 0")
    if bool(args.mp4_enable) and len(_normalize_patients(list(args.mp4_patient or []))) == 0:
        raise ValueError("--mp4-enable requires at least one --mp4-patient.")

    root_dir = Path(__file__).resolve().parents[1]
    test_opt = Path(args.test_opt)
    if not test_opt.is_absolute():
        test_opt = root_dir / test_opt
    if not test_opt.exists():
        raise FileNotFoundError(f"Test config not found: {test_opt}")

    base_cfg = _load_yaml(test_opt)
    models_dir = _resolve_models_dir(root_dir, base_cfg, args.models_dir)
    lock_file = Path(args.lock_file) if args.lock_file else (models_dir / ".async_eval.lock")
    if not lock_file.is_absolute():
        lock_file = root_dir / lock_file
    lock_fh = _acquire_single_flight_lock(lock_file)
    if lock_fh is None:
        print(f"[async-eval] another evaluator is running, trigger ignored. lock={lock_file}")
        return 0

    try:
        if bool(args.once) and not models_dir.exists():
            print(f"[async-eval] models_dir not found, nothing to do: {models_dir}")
            return 0

        state_file = Path(args.state_file) if args.state_file else (models_dir / ".async_eval_state.json")
        if not state_file.is_absolute():
            state_file = root_dir / state_file

        state = _load_state(state_file)
        done = set(str(x) for x in state.get("done", []))
        failed_counts = {str(k): int(v) for k, v in (state.get("failed_counts", {}) or {}).items()}

        print(f"[async-eval] test_opt={test_opt}")
        print(f"[async-eval] models_dir={models_dir}")
        print(f"[async-eval] state_file={state_file}")
        print(f"[async-eval] lock_file={lock_file}")
        print(
            f"[async-eval] every_n_iters={args.every_n_iters} "
            f"backlog_policy={args.backlog_policy} max_pending={args.max_pending}"
        )

        while True:
            state_dirty = False
            if not models_dir.exists():
                print(f"[async-eval] waiting models_dir: {models_dir}")
                if args.once:
                    break
                time.sleep(max(int(args.poll_seconds), 1))
                continue

            ckpts = _list_checkpoints(
                models_dir=models_dir,
                pattern=str(args.pattern),
                include_latest=bool(args.include_latest),
                include_best=bool(args.include_best),
            )
            pending = [p for p in ckpts if _ckpt_key(p) not in done]
            if not pending:
                if args.once:
                    print("[async-eval] no pending checkpoints, exit.")
                    break
                time.sleep(max(int(args.poll_seconds), 1))
                continue

            pending_after_step, skipped_step = _apply_every_n_filter(pending, int(args.every_n_iters))
            if skipped_step:
                print(f"[async-eval] step-filter skipped={len(skipped_step)}")
                if args.step_skip_policy == "mark_done":
                    added = _mark_done(done, skipped_step)
                    if added > 0:
                        print(f"[async-eval] step-filter marked done={added}")
                        state_dirty = True

            selected, dropped = _apply_backlog_policy(pending_after_step, args.backlog_policy, int(args.max_pending))
            if dropped:
                print(f"[async-eval] backlog dropped={len(dropped)} by policy={args.backlog_policy}")
                if args.drop_policy == "mark_done":
                    added = _mark_done(done, dropped)
                    if added > 0:
                        print(f"[async-eval] backlog marked done={added}")
                        state_dirty = True

            if state_dirty:
                state["done"] = sorted(done)
                state["failed_counts"] = failed_counts
                _save_state(state_file, state)

            if not selected:
                if args.once:
                    print("[async-eval] no selected checkpoints after filtering, exit.")
                    break
                time.sleep(max(int(args.poll_seconds), 1))
                continue

            for ckpt in selected:
                key = _ckpt_key(ckpt)
                fail_count = int(failed_counts.get(key, 0))
                if fail_count >= int(args.max_retries_per_ckpt):
                    print(
                        f"[async-eval] retry exhausted for {ckpt.name}: "
                        f"{fail_count}/{args.max_retries_per_ckpt}"
                    )
                    if args.retry_exhausted_policy == "mark_done":
                        if key not in done:
                            done.add(key)
                            state["done"] = sorted(done)
                            state["failed_counts"] = failed_counts
                            _save_state(state_file, state)
                    continue

                print(f"[async-eval] evaluating: {ckpt.name}")
                cfg = _build_eval_yaml(base_cfg, ckpt.resolve(), args.param_key_g)
                with tempfile.TemporaryDirectory(prefix="spect_async_eval_") as td:
                    tmp_opt = Path(td) / "test_async.yml"
                    _save_yaml(tmp_opt, cfg)
                    code = _run_eval(root_dir, args.python, tmp_opt, bool(args.dry_run))
                if code == 0 and bool(args.mp4_enable):
                    code = _run_mp4_for_ckpt(
                        root_dir=root_dir,
                        python_exe=str(args.python),
                        args=args,
                        models_dir=models_dir,
                        ckpt=ckpt,
                        dry_run=bool(args.dry_run),
                    )

                if code == 0:
                    done.add(key)
                    failed_counts.pop(key, None)
                    state["done"] = sorted(done)
                    state["failed_counts"] = failed_counts
                    _save_state(state_file, state)
                    print(f"[async-eval] done: {ckpt.name}")
                else:
                    failed_counts[key] = fail_count + 1
                    state["done"] = sorted(done)
                    state["failed_counts"] = failed_counts
                    _save_state(state_file, state)
                    print(
                        f"[async-eval] failed({code}): {ckpt.name} "
                        f"retry={failed_counts[key]}/{args.max_retries_per_ckpt}"
                    )

            if args.once:
                break

        return 0
    finally:
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            lock_fh.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
