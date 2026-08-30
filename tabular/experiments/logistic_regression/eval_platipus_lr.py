import os
import sys
import glob
import argparse
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

# Add repo root to sys.path
REPO_ROOT = str(Path(__file__).resolve().parents[3])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tabular.dataloader.TabularDataLoader import tabular_train_val_split, load_config, resolve_device, find_best_checkpoint
from tabular.metrics_utils import compute_metrics
from _utils import FocalLoss


def main():
    parser = argparse.ArgumentParser(description="Evaluate PLATIPUS + LogisticRegression on all test cases.")
    parser.add_argument("--test_case", type=str, default=None, help="Optional specific test case name/file to evaluate.")
    args = parser.parse_args()

    cfg = load_config()
    eval_cfg = cfg.get('fast_adaptation_eval', cfg.get('test_cases', {}))
    loss_cfg = eval_cfg.get('focal_loss', {'alpha': 0.25, 'gamma': 2.0})

    data_dir = os.path.join(REPO_ROOT, 'tabular', 'data', 'processed')
    log_dir = os.path.join(REPO_ROOT, 'tabular', 'outputs', 'logs_platipus_lr')
    all_task_files = sorted(glob.glob(os.path.join(data_dir, "*_tasks.pt")))
    test_case_files = [f for f in all_task_files if "dataset_prepro" not in os.path.basename(f)]

    if args.test_case:
        test_case_files = [f for f in test_case_files if args.test_case.lower() in os.path.basename(f).lower()]

    if not test_case_files:
        raise FileNotFoundError(f"No test case .pt files found in: {data_dir}. Run Preprocessing.py first.")

    print(f"Found {len(test_case_files)} test case(s) for evaluation:")
    for f in test_case_files:
        print(f"  - {os.path.basename(f)}")

    # Automatically find best checkpoint epoch based on peak validation accuracy
    best_epoch, best_ckpt_path, best_val_acc = find_best_checkpoint(log_dir)
    acc_str = f" (Peak Val Acc: {best_val_acc:.2f}%)" if best_val_acc is not None else ""
    print(f"\n[AUTO-CHECKPOINT] Automatically selected best model: Epoch {best_epoch}{acc_str}")

    STEPS_PER_DAY = 144
    support_days = eval_cfg.get('support_days', 1)
    k_shot = support_days * STEPS_PER_DAY
    device = resolve_device(eval_cfg.get('device', 'cuda'))

    config = {
        'num_ways': 2,
        'k_shot': k_shot,
        'device': device,
        'network_architecture': 'LogisticRegression',
        'train_val_split_function': tabular_train_val_split,
        'num_inner_updates': eval_cfg.get('num_inner_updates', 5),
        'inner_lr': eval_cfg.get('inner_lr', 0.01),
        'meta_lr': 1e-3,
        'KL_weight': eval_cfg.get('KL_weight', 1e-6),
        'num_models': eval_cfg.get('num_models', 4),
        'first_order': eval_cfg.get('first_order', True),
        'train_flag': False,
        'resume_epoch': best_epoch,
        'num_epochs': best_epoch,
        'num_episodes': 1,
        'num_episodes_per_epoch': 1,
        'minibatch': 1,
        'minibatch_print': 1,
        'logdir': log_dir,
        'loss_function': FocalLoss(alpha=loss_cfg.get('alpha', 0.25), gamma=loss_cfg.get('gamma', 2.0))
    }

    print(f"Initializing PLATIPUS on {device} and loading pre-trained model (Epoch {best_epoch})...")
    from Platipus import Platipus
    platipus = Platipus(config=config)

    dummy_tasks = torch.load(test_case_files[0], weights_only=False)
    dummy_ep = [{
        "x_s": dummy_tasks[0]['x_s'].unsqueeze(0),
        "y_s": dummy_tasks[0]['y_s'].unsqueeze(0),
        "x_q": dummy_tasks[0]['x_q'].unsqueeze(0),
        "y_q": dummy_tasks[0]['y_q'].unsqueeze(0),
        "task_id": int(dummy_tasks[0].get('task_id', 1))
    }]
    model = platipus.load_model(resume_epoch=best_epoch, hyper_net_class=platipus.hyper_net_class, eps_dataloader=dummy_ep)

    for tc_path in test_case_files:
        tc_name = os.path.basename(tc_path)
        tasks = torch.load(tc_path, weights_only=False)
        ep0 = tasks[0]
        x_s, y_s = ep0['x_s'], ep0['y_s']
        x_q, y_q = ep0['x_q'], ep0['y_q']
        task_id = ep0.get('task_id', 1)

        ep = [{
            "x_s": x_s.unsqueeze(0),
            "y_s": y_s.unsqueeze(0),
            "x_q": x_q.unsqueeze(0),
            "y_q": y_q.unsqueeze(0),
            "task_id": int(task_id)
        }]

        split_data = tabular_train_val_split(ep[0])
        x_t = split_data['x_t'].to(device)
        y_t = split_data['y_t'].to(device)
        x_v = split_data['x_v'].to(device)
        y_v = split_data['y_v'].to(device)

        # 1. Fast adaptation returns list of adapted models (phi)
        phi = platipus.adaptation(x=x_t, y=y_t, model=model)

        # 2. Prediction on query set across ensemble of models
        logits_list = platipus.prediction(x=x_v, phi=phi, model=model)
        probs_list = [torch.softmax(l, dim=1) for l in logits_list]
        avg_probs = torch.stack(probs_list).mean(dim=0)

        cls_threshold = float(eval_cfg.get("classification_threshold", 0.5))
        y_pred = (avg_probs[:, 1] >= cls_threshold).long().cpu().numpy()
        y_true = y_v.cpu().numpy()

        compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            test_case=tc_name,
            task_id=task_id,
            model="PLATIPUS",
            architecture="LogisticRegression",
            mode="Fast Adaptation",
            query_size=list(x_q.shape)
        )

        print("\n" + "="*60)
        print(f"EVALUATION: {tc_name} (Task ID {task_id}) [PLATIPUS LR]")
        print(f"Support: {list(x_s.shape)} | Query: {list(x_q.shape)}")
        print("="*60)
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=["Not Smoking (0)", "Smoking (1)"], zero_division=0))
        print("="*60)


if __name__ == "__main__":
    main()
