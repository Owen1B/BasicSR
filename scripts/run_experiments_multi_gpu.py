#!/usr/bin/env python3
"""
SPECT Self-Supervised Denoising 多GPU实验调度器
================================================
使用 rich 库提供美观的终端界面，自动调度实验到多个 GPU

用法:
    python scripts/run_experiments_multi_gpu.py --gpus 0,1
    python scripts/run_experiments_multi_gpu.py --gpus 0,1 --group tier1
    python scripts/run_experiments_multi_gpu.py --gpus 0,1,2,3 --dry-run
    python scripts/run_experiments_multi_gpu.py --gpus 0,1 --skip-existing
    python scripts/run_experiments_multi_gpu.py --gpus 0,1 --debug  # 200步快速测试

依赖:
    pip install rich pyyaml
"""

import os
import sys
import time
import argparse
import subprocess
import threading
import tempfile
import shutil
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from queue import Queue
from collections import defaultdict

# Rich imports
from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TaskProgressColumn
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich import box
from collections import deque
import re

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_BASE = PROJECT_ROOT / "options" / "train" / "spect_selfsup"

console = Console()


# ============================================================
# 实验配置定义
# ============================================================

EXPERIMENT_GROUPS = {
    "baseline": [
        "baseline/n2n_baseline.yml",
        "baseline/n2v_baseline.yml",
        "baseline/n2b_baseline.yml",
        "baseline/n2n_restormer.yml",
    ],
    "tier1": [
        "tier1_method_comparison/n2n_baseline.yml",
        "tier1_method_comparison/n2v_baseline.yml",
        "tier1_method_comparison/n2b_baseline.yml",
        "tier1_method_comparison/noiser2noise_baseline.yml",
    ],
    "tier2_norm": [
        "tier2_ablation/norm/n2n_linear_poisson.yml",
        "tier2_ablation/norm/n2n_linear_mse.yml",
        "tier2_ablation/norm/n2n_linear_l1.yml",
        "tier2_ablation/norm/n2n_linear_charb.yml",
        "tier2_ablation/norm/n2n_anscombe_charb.yml",
        "tier2_ablation/norm/n2n_anscombe_poisson.yml",
    ],
    "tier2_data": [
        "tier2_ablation/data_strategy/n2n_patch64_batch32.yml",
        "tier2_ablation/data_strategy/n2n_patch64_batch16.yml",
        "tier2_ablation/data_strategy/n2n_patch64_batch8.yml",
        "tier2_ablation/data_strategy/n2n_patch128_batch16.yml",
        "tier2_ablation/data_strategy/n2n_patch256_batch8.yml",
        "tier2_ablation/data_strategy/n2n_patch32_batch64.yml",
        "tier2_ablation/data_strategy/n2n_patch64_batch16_aug.yml",
    ],
    "tier2_reg": [
        "tier2_ablation/regularization/n2n_wd_1e4.yml",
        "tier2_ablation/regularization/n2n_wd_0.yml",
        "tier2_ablation/regularization/n2n_wd_1e5.yml",
        "tier2_ablation/regularization/n2n_wd_5e4.yml",
        "tier2_ablation/regularization/n2n_wd_1e3.yml",
    ],
    "tier2_opt": [
        "tier2_ablation/optimization/n2n_adamw_1e4.yml",
        "tier2_ablation/optimization/n2n_adam_1e4.yml",
        "tier2_ablation/optimization/n2n_adamw_5e5.yml",
        "tier2_ablation/optimization/n2n_adamw_5e4.yml",
        "tier2_ablation/optimization/n2n_etamin_1e6.yml",
        "tier2_ablation/optimization/n2n_etamin_1e7.yml",
        "tier2_ablation/optimization/n2n_etamin_1e4.yml",
    ],
    "tier2_net": [
        "tier2_ablation/network/n2n_unetres_nb4.yml",
        "tier2_ablation/network/n2n_unetres_nb2.yml",
        "tier2_ablation/network/n2n_unetres_nb6.yml",
        "tier2_ablation/network/n2n_nc_32_64_128_256.yml",
        "tier2_ablation/network/n2n_nc_96_192_384_768.yml",
    ],
    "tier2_view": [
        "tier2_ablation/view_mode/n2n_single_view_both.yml",
        "tier2_ablation/view_mode/n2n_single_view_anterior.yml",
        "tier2_ablation/view_mode/n2n_single_view_posterior.yml",
    ],
    "tier2_combined": [
        "tier2_ablation/combined/n2n_best_config.yml",
        "tier2_ablation/combined/n2n_original_kair.yml",
        "tier2_ablation/combined/n2n_worst_config.yml",
    ],
    "tier3_n2v": [
        "tier3_other_methods/n2v/n2v_linear_poisson.yml",
        "tier3_other_methods/n2v/n2v_linear_charb.yml",
        "tier3_other_methods/n2v/n2v_anscombe_charb.yml",
        "tier3_other_methods/n2v/n2v_anscombe_poisson.yml",
    ],
    "tier3_n2b": [
        "tier3_other_methods/n2b/n2b_linear_lambda_1_1.yml",
        "tier3_other_methods/n2b/n2b_anscombe_lambda_1_1.yml",
        "tier3_other_methods/n2b/n2b_anscombe_lambda_1_2.yml",
        "tier3_other_methods/n2b/n2b_anscombe_lambda_2_1.yml",
    ],
}

# 组合组
EXPERIMENT_GROUPS["tier2"] = (
    EXPERIMENT_GROUPS["tier2_norm"] +
    EXPERIMENT_GROUPS["tier2_data"] +
    EXPERIMENT_GROUPS["tier2_reg"] +
    EXPERIMENT_GROUPS["tier2_opt"] +
    EXPERIMENT_GROUPS["tier2_net"] +
    EXPERIMENT_GROUPS["tier2_view"] +
    EXPERIMENT_GROUPS["tier2_combined"]
)

EXPERIMENT_GROUPS["tier3"] = (
    EXPERIMENT_GROUPS["tier3_n2v"] +
    EXPERIMENT_GROUPS["tier3_n2b"]
)

EXPERIMENT_GROUPS["all"] = (
    EXPERIMENT_GROUPS["baseline"] +
    EXPERIMENT_GROUPS["tier1"] +
    EXPERIMENT_GROUPS["tier2"] +
    EXPERIMENT_GROUPS["tier3"]
)


# ============================================================
# 数据类
# ============================================================

@dataclass
class Experiment:
    """实验配置"""
    config: str
    name: str = ""
    status: str = "pending"  # pending, running, completed, failed, skipped
    gpu: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    process: Optional[subprocess.Popen] = None
    log_file: Optional[Path] = None
    current_iter: int = 0
    total_iter: int = 0
    current_loss: float = 0.0
    current_psnr: float = 0.0

    def __post_init__(self):
        if not self.name:
            self.name = self.get_exp_name()

    def get_exp_name(self) -> str:
        """从配置文件提取实验名"""
        config_path = CONFIG_BASE / self.config
        if config_path.exists():
            with open(config_path) as f:
                for line in f:
                    if line.startswith("name:"):
                        return line.split(":", 1)[1].strip()
        return Path(self.config).stem

    def check_existing(self) -> bool:
        """检查是否已有检查点"""
        state_file = PROJECT_ROOT / "experiments" / self.name / "training_states" / "latest.state"
        return state_file.exists()

    @property
    def duration(self) -> Optional[timedelta]:
        if self.start_time:
            end = self.end_time or datetime.now()
            return end - self.start_time
        return None

    @property
    def duration_str(self) -> str:
        if self.duration:
            total_seconds = int(self.duration.total_seconds())
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return "--:--:--"


@dataclass
class GPUWorker:
    """GPU工作器"""
    gpu_id: int
    current_experiment: Optional[Experiment] = None
    total_completed: int = 0
    total_failed: int = 0

    @property
    def is_busy(self) -> bool:
        return self.current_experiment is not None

    @property
    def status(self) -> str:
        if self.current_experiment:
            return f"🔄 {self.current_experiment.name[:30]}"
        return "⏳ 空闲"


# ============================================================
# 调度器
# ============================================================

class ExperimentScheduler:
    """多GPU实验调度器"""

    def __init__(self, gpu_ids: List[int], experiments: List[str],
                 skip_existing: bool = False, dry_run: bool = False,
                 debug_mode: bool = False, debug_iters: int = 200):
        self.gpu_ids = gpu_ids
        self.skip_existing = skip_existing
        self.dry_run = dry_run
        self.debug_mode = debug_mode
        self.debug_iters = debug_iters

        # Debug 模式临时目录
        self.temp_dir = None
        if self.debug_mode:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="spect_debug_"))

        # 创建实验队列
        self.experiments = [Experiment(config=cfg) for cfg in experiments]
        self.pending_queue: Queue = Queue()
        self.completed: List[Experiment] = []
        self.failed: List[Experiment] = []
        self.skipped: List[Experiment] = []

        # 创建GPU工作器
        self.workers = {gpu_id: GPUWorker(gpu_id=gpu_id) for gpu_id in gpu_ids}

        # 控制标志
        self.running = False
        self.lock = threading.Lock()

        # 日志缓冲区 (每个 GPU 一个)
        self.log_buffers = {gpu_id: deque(maxlen=15) for gpu_id in gpu_ids}

        # 初始化队列
        self._init_queue()

    def _init_queue(self):
        """初始化实验队列"""
        for exp in self.experiments:
            if self.skip_existing and exp.check_existing():
                exp.status = "skipped"
                self.skipped.append(exp)
            else:
                self.pending_queue.put(exp)

    def _create_debug_config(self, original_config: Path) -> Path:
        """创建 debug 模式的临时配置文件"""
        if not original_config.exists():
            raise FileNotFoundError(f"Config not found: {original_config}")

        # 读取原始配置
        with open(original_config, 'r') as f:
            config = yaml.safe_load(f)

        # 修改为 debug 模式
        original_name = config.get('name', original_config.stem)
        config['name'] = f"debug_{original_name}"

        # 修改训练参数
        if 'train' in config:
            config['train']['total_iter'] = self.debug_iters

        # 注意: 不修改 val_freq，BasicSR 的 --debug 参数会自动设为 8

        # 禁用 wandb
        if 'logger' in config and 'wandb' in config['logger']:
            config['logger']['wandb']['project'] = None

        # 保存临时配置
        temp_config = self.temp_dir / f"debug_{original_config.stem}.yml"
        with open(temp_config, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        return temp_config

    def _run_experiment(self, exp: Experiment, gpu_id: int):
        """运行单个实验"""
        exp.gpu = gpu_id
        exp.status = "running"
        exp.start_time = datetime.now()

        original_config_path = CONFIG_BASE / exp.config

        # Debug 模式：创建临时配置
        if self.debug_mode:
            try:
                config_path = self._create_debug_config(original_config_path)
                exp_name = f"debug_{exp.name}"
            except Exception as e:
                exp.status = "failed"
                exp.end_time = datetime.now()
                console.print(f"[red]Failed to create debug config for {exp.name}: {e}[/]")
                return False
        else:
            config_path = original_config_path
            exp_name = exp.name

        log_dir = PROJECT_ROOT / "experiments" / exp_name
        log_dir.mkdir(parents=True, exist_ok=True)
        exp.log_file = log_dir / f"scheduler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONPATH"] = f".:{env.get('PYTHONPATH', '')}"

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "basicsr" / "train.py"),
            "-opt", str(config_path),
        ]

        # Debug 模式添加 --debug 参数
        if self.debug_mode:
            cmd.append("--debug")
        else:
            cmd.append("--auto_resume")

        if self.dry_run:
            console.print(f"[yellow][DRY RUN] GPU {gpu_id}: {' '.join(cmd)}[/]")
            time.sleep(2)  # 模拟运行
            exp.status = "completed"
            exp.end_time = datetime.now()
            return True

        try:
            with open(exp.log_file, "w") as log:
                exp.process = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    universal_newlines=True
                )
                # 实时读取输出并写入文件
                for line in exp.process.stdout:
                    log.write(line)
                    log.flush()
                    # 解析日志更新进度
                    self._parse_log_line(exp, line, gpu_id)

                exp.process.wait()

            if exp.process.returncode == 0:
                exp.status = "completed"
                return True
            else:
                exp.status = "failed"
                self._show_error_log(exp.log_file, exp.name)
                return False
        except Exception as e:
            exp.status = "failed"
            with self.lock:
                self.log_buffers[gpu_id].append(f"[red]Error: {e}[/]")
            return False
        finally:
            exp.end_time = datetime.now()

    def _parse_log_line(self, exp: Experiment, line: str, gpu_id: int):
        """解析日志行，提取进度信息"""
        line = line.strip()
        if not line:
            return

        # 添加到日志缓冲区
        with self.lock:
            # 过滤掉太长的行和配置输出
            if len(line) < 200 and not line.startswith('  '):
                self.log_buffers[gpu_id].append(line)

        # === 解析 tqdm 进度条格式 ===
        # 格式: Train:  93%|...| 28/30 [00:16<00:01, 1.56iter/s, epoch=2, l_pix=1.58, lr=0.0001]
        tqdm_match = re.search(r'Train:\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)', line)
        if tqdm_match:
            exp.current_iter = int(tqdm_match.group(1))
            exp.total_iter = int(tqdm_match.group(2))

        # tqdm 格式的 loss: l_pix=1.58
        tqdm_loss_match = re.search(r'l_pix=([\d.e+-]+)', line)
        if tqdm_loss_match:
            try:
                exp.current_loss = float(tqdm_loss_match.group(1))
            except:
                pass

        # === 解析传统 BasicSR 日志格式 ===
        # 格式: [exp..][epoch:  1, iter:     100, lr:(1.000e-04,)] l_pix: 1.6904e+00
        iter_match = re.search(r'iter:\s*(\d+)', line)
        if iter_match and not tqdm_match:  # 如果没有匹配 tqdm 格式才用这个
            exp.current_iter = int(iter_match.group(1))

        # 解析 total_iter (从 Total epochs: X; iters: Y)
        total_match = re.search(r'iters:\s*(\d+)', line)
        if total_match and 'Total' in line:
            exp.total_iter = int(total_match.group(1))

        # 解析传统格式 loss: l_pix: 1.6904e+00
        loss_match = re.search(r'l_pix:\s*([\d.e+-]+)', line)
        if loss_match and not tqdm_loss_match:  # 如果没有匹配 tqdm 格式才用这个
            try:
                exp.current_loss = float(loss_match.group(1))
            except:
                pass

        # 解析验证 PSNR (两种格式都一样)
        psnr_match = re.search(r'psnr_cnt_ema:\s*([\d.]+)', line)
        if psnr_match:
            try:
                exp.current_psnr = float(psnr_match.group(1))
            except:
                pass

    def _show_error_log(self, log_file: Path, exp_name: str):
        """显示错误日志的最后几行"""
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                last_lines = lines[-20:] if len(lines) > 20 else lines
                error_lines = [l for l in last_lines if 'error' in l.lower() or 'exception' in l.lower() or 'traceback' in l.lower()]
                if error_lines:
                    console.print(f"[red]Error in {exp_name}:[/]")
                    for line in error_lines[:5]:
                        console.print(f"  [dim]{line.strip()}[/]")
        except Exception:
            pass

    def _worker_thread(self, gpu_id: int):
        """GPU工作线程"""
        worker = self.workers[gpu_id]

        while self.running:
            try:
                # 尝试获取下一个任务
                exp = self.pending_queue.get(timeout=1)

                with self.lock:
                    worker.current_experiment = exp

                # 运行实验
                success = self._run_experiment(exp, gpu_id)

                with self.lock:
                    if success:
                        self.completed.append(exp)
                        worker.total_completed += 1
                    else:
                        self.failed.append(exp)
                        worker.total_failed += 1
                    worker.current_experiment = None

                self.pending_queue.task_done()

            except:
                # 队列超时或空
                continue

    def _create_status_table(self) -> Table:
        """创建状态表格"""
        table = Table(title="🖥️ GPU 状态", box=box.ROUNDED)
        table.add_column("GPU", style="cyan", width=6)
        table.add_column("状态", width=40)
        table.add_column("完成", style="green", width=6)
        table.add_column("失败", style="red", width=6)

        for gpu_id, worker in self.workers.items():
            table.add_row(
                str(gpu_id),
                worker.status,
                str(worker.total_completed),
                str(worker.total_failed)
            )

        return table

    def _create_progress_table(self) -> Table:
        """创建进度表格"""
        total = len(self.experiments)
        completed = len(self.completed)
        failed = len(self.failed)
        skipped = len(self.skipped)
        running = sum(1 for w in self.workers.values() if w.is_busy)
        pending = self.pending_queue.qsize()

        table = Table(title="📊 实验进度", box=box.ROUNDED)
        table.add_column("类别", style="bold")
        table.add_column("数量", justify="right")
        table.add_column("进度条", width=30)

        # 总进度
        done = completed + failed + skipped
        pct = done / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        table.add_row("总计", f"{done}/{total}", f"[green]{bar}[/] {pct:.1f}%")

        table.add_row("✅ 完成", f"[green]{completed}[/]", "")
        table.add_row("❌ 失败", f"[red]{failed}[/]", "")
        table.add_row("⏭️ 跳过", f"[yellow]{skipped}[/]", "")
        table.add_row("🔄 运行中", f"[blue]{running}[/]", "")
        table.add_row("⏳ 等待", f"[dim]{pending}[/]", "")

        return table

    def _create_running_table(self) -> Table:
        """创建运行中实验表格"""
        table = Table(title="🔄 运行中的实验", box=box.ROUNDED, expand=True)
        table.add_column("GPU", style="cyan", width=4)
        table.add_column("实验名", width=28)
        table.add_column("进度", width=20)
        table.add_column("Loss", width=10)
        table.add_column("PSNR", width=8)
        table.add_column("耗时", width=10)

        for gpu_id, worker in self.workers.items():
            if worker.current_experiment:
                exp = worker.current_experiment
                # 进度条
                if exp.total_iter > 0:
                    pct = exp.current_iter / exp.total_iter
                    bar_len = 10
                    filled = int(pct * bar_len)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    progress = f"{bar} {exp.current_iter}/{exp.total_iter}"
                else:
                    progress = f"iter: {exp.current_iter}"

                table.add_row(
                    str(gpu_id),
                    exp.name[:26] + ".." if len(exp.name) > 28 else exp.name,
                    progress,
                    f"{exp.current_loss:.2e}" if exp.current_loss > 0 else "-",
                    f"{exp.current_psnr:.2f}" if exp.current_psnr > 0 else "-",
                    exp.duration_str
                )
            else:
                table.add_row(str(gpu_id), "[dim]空闲[/]", "-", "-", "-", "-")

        return table

    def _create_log_panel(self) -> Panel:
        """创建日志面板"""
        log_lines = []
        for gpu_id in self.gpu_ids:
            if self.log_buffers[gpu_id]:
                worker = self.workers[gpu_id]
                exp_name = worker.current_experiment.name[:20] if worker.current_experiment else "idle"
                log_lines.append(f"[bold cyan]── GPU {gpu_id} ({exp_name}) ──[/]")
                for line in list(self.log_buffers[gpu_id])[-5:]:  # 每个GPU最多显示5行
                    # 截断太长的行
                    if len(line) > 100:
                        line = line[:97] + "..."
                    log_lines.append(f"  {line}")

        if not log_lines:
            log_lines = ["[dim]等待实验开始...[/]"]

        return Panel(
            "\n".join(log_lines),
            title="📜 实时日志",
            box=box.ROUNDED,
            height=20
        )

    def _create_layout(self) -> Layout:
        """创建布局"""
        layout = Layout()

        # 标题
        mode = "🔧 DEBUG" if self.debug_mode else "🚀 正式"
        title = Text(f"🧪 SPECT Self-Supervised Denoising 多GPU调度器 [{mode}]", style="bold magenta")
        title_panel = Panel(title, box=box.DOUBLE)

        # 组装
        layout.split_column(
            Layout(title_panel, size=3),
            Layout(name="top", size=12),
            Layout(name="bottom")
        )

        # 上部分：状态 + 进度 + 运行中实验
        layout["top"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=2)
        )

        layout["left"].split_column(
            Layout(self._create_status_table(), name="gpu"),
            Layout(self._create_progress_table(), name="progress")
        )

        layout["right"].update(self._create_running_table())

        # 下部分：日志
        layout["bottom"].update(self._create_log_panel())

        return layout

    def run(self):
        """运行调度器"""
        mode_str = "[bold yellow]🔧 DEBUG 模式[/]" if self.debug_mode else "[bold green]🚀 正式模式[/]"
        debug_info = f"\n迭代数: {self.debug_iters}\n临时目录: {self.temp_dir}" if self.debug_mode else ""

        console.print(Panel.fit(
            f"{mode_str}\n"
            f"GPU: {self.gpu_ids}\n"
            f"实验数: {len(self.experiments)}\n"
            f"跳过已存在: {self.skip_existing}\n"
            f"Dry Run: {self.dry_run}"
            f"{debug_info}",
            title="配置"
        ))

        if self.pending_queue.empty():
            console.print("[yellow]没有待运行的实验！[/]")
            return

        self.running = True

        # 启动工作线程
        threads = []
        for gpu_id in self.gpu_ids:
            t = threading.Thread(target=self._worker_thread, args=(gpu_id,), daemon=True)
            t.start()
            threads.append(t)

        # 实时显示
        try:
            with Live(self._create_layout(), refresh_per_second=1, console=console) as live:
                while self.running:
                    # 检查是否所有任务完成
                    if self.pending_queue.empty() and not any(w.is_busy for w in self.workers.values()):
                        self.running = False
                        break

                    live.update(self._create_layout())
                    time.sleep(0.5)
        except KeyboardInterrupt:
            console.print("\n[yellow]用户中断，正在停止...[/]")
            self.running = False

# 等待线程结束
        for t in threads:
            t.join(timeout=5)

        # 清理 debug 临时目录
        if self.debug_mode and self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                console.print(f"[dim]已清理临时目录: {self.temp_dir}[/]")
            except:
                pass

        # 打印总结
        self._print_summary()

    def _print_summary(self):
        """打印运行总结"""
        console.print("\n")
        console.print(Panel.fit(
            f"[bold]实验运行总结[/]\n\n"
            f"✅ 完成: [green]{len(self.completed)}[/]\n"
            f"❌ 失败: [red]{len(self.failed)}[/]\n"
            f"⏭️ 跳过: [yellow]{len(self.skipped)}[/]\n",
            title="📊 总结"
        ))

        if self.failed:
            console.print("\n[red]失败的实验:[/]")
            for exp in self.failed:
                console.print(f"  - {exp.name}")
                if exp.log_file:
                    console.print(f"    日志: {exp.log_file}")

        if self.completed:
            console.print("\n[green]完成的实验:[/]")
            for exp in self.completed:
                console.print(f"  - {exp.name} ({exp.duration_str})")


# ============================================================
# 命令行入口
# ============================================================

def list_groups():
    """列出所有实验组"""
    table = Table(title="📋 实验组列表", box=box.ROUNDED)
    table.add_column("组名", style="cyan")
    table.add_column("实验数", justify="right")
    table.add_column("说明")

    descriptions = {
        "baseline": "基础验证 (N2N, N2V, N2B, Restormer)",
        "tier1": "方法对比 (5种自监督方法)",
        "tier2": "所有 Tier2 消融实验",
        "tier2_norm": "归一化/损失函数消融",
        "tier2_data": "数据策略消融",
        "tier2_reg": "正则化消融",
        "tier2_opt": "优化器消融",
        "tier2_net": "网络结构消融",
        "tier2_view": "视角模式消融",
        "tier2_combined": "组合配置",
        "tier3": "所有 Tier3 实验",
        "tier3_n2v": "N2V 变体",
        "tier3_n2b": "N2B 变体",
        "all": "全部实验",
    }

    for name, configs in EXPERIMENT_GROUPS.items():
        if name not in ["tier2", "tier3", "all"]:  # 跳过组合组，单独显示
            table.add_row(name, str(len(configs)), descriptions.get(name, ""))

    # 组合组
    table.add_row("─" * 10, "─" * 5, "─" * 30)
    for name in ["tier2", "tier3", "all"]:
        table.add_row(
            f"[bold]{name}[/]",
            f"[bold]{len(EXPERIMENT_GROUPS[name])}[/]",
            descriptions.get(name, "")
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="SPECT Self-Supervised Denoising 多GPU实验调度器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 正式运行
  python scripts/run_experiments_multi_gpu.py --gpus 0,1 --group all
  python scripts/run_experiments_multi_gpu.py --gpus 0,1,2,3 --group tier2
  python scripts/run_experiments_multi_gpu.py --gpus 0,1 --skip-existing

  # Debug 模式 (快速测试配置和显存)
  python scripts/run_experiments_multi_gpu.py --gpus 0,1 --debug
  python scripts/run_experiments_multi_gpu.py --gpus 0,1 --debug --debug-iters 100

  # 查看实验组
  python scripts/run_experiments_multi_gpu.py --list
        """
    )

    parser.add_argument("--gpus", type=str, default="0,1",
                        help="GPU IDs，逗号分隔 (默认: 0,1)")
    parser.add_argument("--group", type=str, default="all",
                        choices=list(EXPERIMENT_GROUPS.keys()),
                        help="实验组 (默认: all)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过已有检查点的实验")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印命令不执行")
    parser.add_argument("--debug", action="store_true",
                        help="Debug 模式: 200步快速测试配置和显存")
    parser.add_argument("--debug-iters", type=int, default=200,
                        help="Debug 模式的迭代数 (默认: 200)")
    parser.add_argument("--list", action="store_true",
                        help="列出所有实验组")

    args = parser.parse_args()

    if args.list:
        list_groups()
        return

    # 解析 GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]

    # 获取实验列表
    experiments = EXPERIMENT_GROUPS.get(args.group, [])

    if not experiments:
        console.print(f"[red]未找到实验组: {args.group}[/]")
        return

    # 创建调度器
    scheduler = ExperimentScheduler(
        gpu_ids=gpu_ids,
        experiments=experiments,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
        debug_mode=args.debug,
        debug_iters=args.debug_iters
    )

    # 运行
    scheduler.run()


if __name__ == "__main__":
    main()

