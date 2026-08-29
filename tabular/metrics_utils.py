"""
tabular/metrics_utils.py
------------------------
Utility functions for extracting, serializing, and aggregating structured evaluation metrics.
"""

import os
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

REPO_ROOT = str(Path(__file__).resolve().parents[1])
RESULTS_JSON_PATH = os.path.join(REPO_ROOT, "tabular", "outputs", "pipeline_results.json")


def compute_metrics(y_true, y_pred, test_case: str, task_id: int, model: str, architecture: str, mode: str, query_size: list):
    """Computes structured dictionary of classification metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=["Not Smoking (0)", "Smoking (1)"], output_dict=True, zero_division=0)

    accuracy = float(report.get("accuracy", 0.0))
    c0 = report.get("Not Smoking (0)", {})
    c1 = report.get("Smoking (1)", {})
    macro = report.get("macro avg", {})
    weighted = report.get("weighted avg", {})

    record = {
        "test_case": os.path.basename(test_case),
        "task_id": int(task_id),
        "model": model,
        "architecture": architecture,
        "mode": mode,  # "Fast Adaptation" or "Cold Start"
        "query_size": query_size,
        "accuracy": round(accuracy * 100, 2),
        "macro_f1": round(float(macro.get("f1-score", 0.0)) * 100, 2),
        "weighted_f1": round(float(weighted.get("f1-score", 0.0)) * 100, 2),
        "class_0": {
            "precision": round(float(c0.get("precision", 0.0)) * 100, 2),
            "recall": round(float(c0.get("recall", 0.0)) * 100, 2),
            "f1": round(float(c0.get("f1-score", 0.0)) * 100, 2),
            "support": int(c0.get("support", 0))
        },
        "class_1": {
            "precision": round(float(c1.get("precision", 0.0)) * 100, 2),
            "recall": round(float(c1.get("recall", 0.0)) * 100, 2),
            "f1": round(float(c1.get("f1-score", 0.0)) * 100, 2),
            "support": int(c1.get("support", 0))
        },
        "confusion_matrix": cm
    }

    # Print standard delimiter for runner to parse
    print("\n__METRICS_JSON_START__")
    print(json.dumps(record))
    print("__METRICS_JSON_END__\n")

    # Persist to pipeline_results.json
    save_metric_record(record)
    return record


def save_metric_record(record: dict):
    """Appends/updates a metric record in pipeline_results.json."""
    os.makedirs(os.path.dirname(RESULTS_JSON_PATH), exist_ok=True)
    
    current_data = {"tasks_status": [], "metrics": []}
    if os.path.exists(RESULTS_JSON_PATH):
        try:
            with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
                current_data = json.load(f)
        except Exception:
            current_data = {"tasks_status": [], "metrics": []}

    metrics = current_data.get("metrics", [])
    
    # Check if duplicate exists (same test_case, model, architecture, mode, task_id)
    key = (record["test_case"], record["model"], record["architecture"], record["mode"], record["task_id"])
    updated = False
    for idx, m in enumerate(metrics):
        m_key = (m.get("test_case"), m.get("model"), m.get("architecture"), m.get("mode"), m.get("task_id"))
        if key == m_key:
            metrics[idx] = record
            updated = True
            break
    
    if not updated:
        metrics.append(record)

    current_data["metrics"] = metrics
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(current_data, f, indent=2)


def clear_all_metrics():
    """Clears all stored metrics and task status records."""
    os.makedirs(os.path.dirname(RESULTS_JSON_PATH), exist_ok=True)
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"tasks_status": [], "metrics": []}, f, indent=2)
