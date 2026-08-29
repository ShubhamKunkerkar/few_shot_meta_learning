"""
run_pipeline.py
---------------
Master pipeline orchestrator that executes the tabular meta-learning workflow
in strict chronological order as specified in pipeline_order.json:
  1. Preprocessing (Raw CSVs -> standardized .pt episodic tasks)
  2. Meta-Training (FcNet & Logistic Regression models)
  3. Fast Adaptation Evaluation (Day 1 Support -> Days 2-3 Query)
  4. Cold-Start / Zero-Shot Evaluation (Days 2-3 Query directly)

Extracts structured metrics, task completion messages, and aggregates results
into tabular/outputs/pipeline_results.json while printing comparison tables.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

RESULTS_JSON_PATH = os.path.join(REPO_ROOT, "tabular", "outputs", "pipeline_results.json")


def get_python_executable():
    """Returns the python executable containing torch, checking virtualenvs if needed."""
    try:
        import torch
        return sys.executable
    except ImportError:
        pass

    candidates = [
        # Windows conda paths
        os.path.expanduser(r"~\.conda\envs\few_shot_meta_learning\python.exe"),
        os.path.expanduser(r"~\anaconda3\envs\few_shot_meta_learning\python.exe"),
        os.path.expanduser(r"~\miniconda3\envs\few_shot_meta_learning\python.exe"),
        # macOS / Linux conda paths
        os.path.expanduser("~/.conda/envs/few_shot_meta_learning/bin/python"),
        os.path.expanduser("~/anaconda3/envs/few_shot_meta_learning/bin/python"),
        os.path.expanduser("~/miniconda3/envs/few_shot_meta_learning/bin/python"),
        os.path.expanduser("~/miniforge3/envs/few_shot_meta_learning/bin/python"),
        os.path.expanduser("~/opt/anaconda3/envs/few_shot_meta_learning/bin/python"),
        "/opt/homebrew/Caskroom/miniforge/base/envs/few_shot_meta_learning/bin/python",
        "/opt/homebrew/anaconda3/envs/few_shot_meta_learning/bin/python",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return sys.executable


def load_pipeline_config(config_path: str) -> Dict[str, Any]:
    """Loads the pipeline order configuration JSON."""
    if not os.path.isabs(config_path):
        config_path = os.path.join(REPO_ROOT, config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Pipeline config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_duration(seconds: float) -> str:
    """Formats duration in seconds to a human-readable string."""
    mins, secs = divmod(int(seconds), 60)
    if mins > 0:
        return f"{mins}m {secs}s"
    return f"{seconds:.2f}s"


def print_comparison_table(metrics: List[Dict[str, Any]]):
    """Prints a structured ASCII side-by-side comparison table (Adapted vs. Cooldown) across test cases."""
    if not metrics:
        return

    # Group by test case
    test_cases = sorted(list(set(m["test_case"] for m in metrics)))

    print("\n" + "=" * 148)
    print(" " * 52 + "MULTI-MODEL EVALUATION COMPARISON TABLE")
    print("=" * 148)

    for tc in test_cases:
        tc_metrics = [m for m in metrics if m["test_case"] == tc]
        print(f"\n[+] TEST CASE: {tc}")
        print("-" * 148)
        print(f"{'Model':<9} | {'Architecture':<18} | {'------- ADAPTED (FAST ADAPTATION) -------':<55} | {'------ COOLDOWN (COLD START / ZERO SHOT) ------':<55}")
        print(f"{'':<9} | {'':<18} | {'Acc':<7} | {'MacroF1':<8} | {'C0 Prec/Rec':<16} | {'C1 Prec/Rec':<16} | {'Acc':<7} | {'MacroF1':<8} | {'C0 Prec/Rec':<16} | {'C1 Prec/Rec':<16}")
        print("-" * 148)

        # Pair by (model, architecture)
        pairs: Dict[Tuple[str, str], Dict[str, Optional[Dict[str, Any]]]] = {}
        for m in tc_metrics:
            key = (m.get("model", ""), m.get("architecture", ""))
            if key not in pairs:
                pairs[key] = {"adapted": None, "cold": None}
            mode = m.get("mode", "").lower()
            if "fast" in mode or "adapt" in mode:
                pairs[key]["adapted"] = m
            else:
                pairs[key]["cold"] = m

        for (model_name, arch_name), data in sorted(pairs.items()):
            ad = data["adapted"]
            cd = data["cold"]

            def fmt_cols(m_dict: Optional[Dict[str, Any]]) -> str:
                if not m_dict:
                    return f"{'-':<7} | {'-':<8} | {'-':<16} | {'-':<16}"
                acc = f"{m_dict.get('accuracy', 0.0):.1f}%"
                mf1 = f"{m_dict.get('macro_f1', 0.0):.1f}%"
                c0 = m_dict.get("class_0", {})
                c1 = m_dict.get("class_1", {})
                c0_pr = f"{c0.get('precision', 0.0):.1f}% / {c0.get('recall', 0.0):.1f}%"
                c1_pr = f"{c1.get('precision', 0.0):.1f}% / {c1.get('recall', 0.0):.1f}%"
                return f"{acc:<7} | {mf1:<8} | {c0_pr:<16} | {c1_pr:<16}"

            ad_str = fmt_cols(ad)
            cd_str = fmt_cols(cd)
            print(f"{model_name:<9} | {arch_name:<18} | {ad_str} | {cd_str}")

        print("-" * 148)
    print("=" * 148 + "\n")


def run_pipeline(
    config_path: str = "tabular/pipeline_order.json",
    target_stage: str = "all",
    target_arch: str = None,
    target_model: str = None,
    dry_run: bool = False,
    stop_on_error: bool = True
):
    cfg = load_pipeline_config(config_path)
    stages = cfg.get("pipeline_stages", [])

    print("\n" + "=" * 80)
    print(" " * 22 + "TABULAR META-LEARNING PIPELINE RUNNER")
    print("=" * 80)
    print(f"Repository Root : {REPO_ROOT}")
    print(f"Config File     : {config_path}")
    print(f"Target Stage    : {target_stage}")
    print(f"Architecture    : {target_arch or 'All'}")
    print(f"Model Filter    : {target_model or 'All'}")
    print(f"Mode            : {'DRY RUN (Preview Only)' if dry_run else 'LIVE EXECUTION'}")
    print(f"Started At      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    overall_start_time = time.time()
    execution_tasks = []

    # Flatten and filter scripts to run
    tasks_to_run = []
    for stage_idx, stage in enumerate(stages, 1):
        stage_id = stage.get("stage_id", f"stage_{stage_idx}")
        stage_name = stage.get("stage_name", stage_id)

        # Filter by stage
        if target_stage.lower() != "all" and target_stage.lower() not in stage_id.lower() and target_stage.lower() not in stage_name.lower():
            continue

        for script_info in stage.get("scripts", []):
            if not script_info.get("enabled", True):
                continue

            arch = script_info.get("architecture")
            model = script_info.get("model")

            # Filter by architecture
            if target_arch and arch and target_arch.lower() not in arch.lower():
                continue

            # Filter by model
            if target_model and model and target_model.lower() not in model.lower():
                continue

            tasks_to_run.append({
                "stage_num": stage_idx,
                "total_stages": len(stages),
                "stage_name": stage_name,
                "stage_id": stage_id,
                "script_info": script_info
            })

    if not tasks_to_run:
        print("No scripts matched the specified filter criteria.")
        return

    print(f"Found {len(tasks_to_run)} script(s) to execute in order:\n")
    for i, t in enumerate(tasks_to_run, 1):
        s_info = t["script_info"]
        print(f"  {i:2d}. [{t['stage_name']}] {s_info['name']} -> {s_info['path']}")
    print("\n" + "-" * 80 + "\n")

    if dry_run:
        print("Dry run completed. No scripts were executed.")
        return

    # Load existing metrics cache
    collected_metrics = []
    if os.path.exists(RESULTS_JSON_PATH):
        try:
            with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
                saved_json = json.load(f)
                collected_metrics = saved_json.get("metrics", [])
        except Exception:
            collected_metrics = []

    # Execute tasks sequentially
    for i, t in enumerate(tasks_to_run, 1):
        s_info = t["script_info"]
        script_rel_path = s_info["path"]
        script_full_path = os.path.join(REPO_ROOT, script_rel_path)

        if not os.path.exists(script_full_path):
            error_msg = f"Script file not found: {script_full_path}"
            print(f"\n[ERROR] {error_msg}")
            execution_tasks.append({
                "name": s_info["name"],
                "path": script_rel_path,
                "stage": t["stage_name"],
                "status": "FAILED (File Not Found)",
                "duration": "0s",
                "message": error_msg
            })
            if stop_on_error:
                break
            continue

        print("\n" + "=" * 80)
        print(f"[{i}/{len(tasks_to_run)}] {t['stage_name']}")
        print(f"Executing : {s_info['name']} ({script_rel_path})")
        print(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80 + "\n")

        py_exec = get_python_executable()
        cmd = [py_exec, "-u", script_full_path]
        step_start = time.time()
        success = False

        try:
            # Stream output live and capture metrics JSON
            proc = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            in_metric_block = False
            metric_json_lines = []

            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()

                stripped = line.strip()
                if stripped == "__METRICS_JSON_START__":
                    in_metric_block = True
                    metric_json_lines = []
                elif stripped == "__METRICS_JSON_END__":
                    in_metric_block = False
                    if metric_json_lines:
                        try:
                            record = json.loads("".join(metric_json_lines))
                            # update or append
                            key = (record["test_case"], record["model"], record["architecture"], record["mode"], record["task_id"])
                            upd = False
                            for midx, m in enumerate(collected_metrics):
                                if key == (m.get("test_case"), m.get("model"), m.get("architecture"), m.get("mode"), m.get("task_id")):
                                    collected_metrics[midx] = record
                                    upd = True
                                    break
                            if not upd:
                                collected_metrics.append(record)
                        except Exception:
                            pass
                elif in_metric_block:
                    metric_json_lines.append(line)

            proc.wait()
            success = (proc.returncode == 0)
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Script failed with return code {e.returncode}: {script_rel_path}")
            success = False
        except Exception as e:
            print(f"\n[ERROR] Unexpected execution exception: {e}")
            success = False

        elapsed = time.time() - step_start
        status_str = "SUCCESS" if success else "FAILED"
        msg = f"{s_info['name']} completed in {format_duration(elapsed)}" if success else f"{s_info['name']} failed (exit code {proc.returncode if 'proc' in locals() else 1})"
        print(f"\n>> Task Result: {status_str} | {msg}\n")

        execution_tasks.append({
            "name": s_info["name"],
            "path": script_rel_path,
            "stage": t["stage_name"],
            "status": status_str,
            "duration": format_duration(elapsed),
            "message": msg
        })

        # Save snapshot
        os.makedirs(os.path.dirname(RESULTS_JSON_PATH), exist_ok=True)
        with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "tasks_status": execution_tasks,
                "metrics": collected_metrics,
                "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)

        if not success and stop_on_error:
            print(f"[PIPELINE ABORTED] Pipeline halted due to failure in: {s_info['name']}")
            break

    # Final Summary Report
    total_elapsed = time.time() - overall_start_time
    print("\n" + "=" * 80)
    print(" " * 28 + "PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"{'#':<3} | {'Task Name':<38} | {'Status':<10} | {'Duration':<10} | {'Message'}")
    print("-" * 80)

    all_passed = True
    for idx, log in enumerate(execution_tasks, 1):
        status = log["status"]
        if "FAILED" in status:
            all_passed = False
        print(f"{idx:<3} | {log['name']:<38} | {status:<10} | {log['duration']:<10} | {log['message']}")

    print("-" * 80)
    print(f"Total Pipeline Runtime: {format_duration(total_elapsed)}")
    print(f"Final Status: {'ALL STAGES PASSED SUCCESSFULLY' if all_passed else 'SOME STAGES FAILED'}")
    print("=" * 80 + "\n")

    # Print Full Comparison Table
    if collected_metrics:
        print_comparison_table(collected_metrics)


def main():
    parser = argparse.ArgumentParser(description="Run full Few-Shot Tabular Meta-Learning Pipeline with structured metrics output.")
    parser.add_argument("--config", type=str, default="tabular/pipeline_order.json",
                        help="Path to pipeline_order.json")
    parser.add_argument("--stage", type=str, default="all",
                        help="Target stage to run: preprocessing, training, fast_adaptation, cold_start, or all.")
    parser.add_argument("--architecture", type=str, default=None,
                        help="Filter by architecture: FcNet or LogisticRegression.")
    parser.add_argument("--model", type=str, default=None,
                        help="Filter by model algorithm: MAML, PLATIPUS, or VAMPIRE.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Preview execution sequence without running commands.")
    parser.add_argument("--continue_on_error", action="store_true",
                        help="Continue running remaining scripts even if one fails.")

    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        target_stage=args.stage,
        target_arch=args.architecture,
        target_model=args.model,
        dry_run=args.dry_run,
        stop_on_error=not args.continue_on_error
    )


if __name__ == "__main__":
    main()
