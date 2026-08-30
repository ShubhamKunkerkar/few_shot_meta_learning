import sys
import os
import glob
import argparse
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = str(Path(__file__).resolve().parents[3])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from Bmaml import Bmaml
from tabular.dataloader.TabularDataLoader import get_tabular_dataloader, tabular_train_val_split, load_config, resolve_device, find_best_checkpoint
from _utils import FocalLoss
from tabular.metrics_utils import compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate BMAML + FcNet on all test cases.")
    parser.add_argument("--test_case", type=str, default=None, help="Optional specific test case name/file to evaluate.")
    args = parser.parse_args()

    cfg = load_config()
    eval_cfg = cfg.get('fast_adaptation_eval', cfg.get('test_cases', {}))
    fcnet_cfg = cfg.get('fcnet', {})
    loss_cfg = eval_cfg.get('focal_loss', {'alpha': 0.25, 'gamma': 2.0})

    data_dir = os.path.join(REPO_ROOT, 'tabular', 'data', 'processed')
    log_dir = os.path.join(REPO_ROOT, 'tabular', 'outputs', 'logs_bmaml_fcnet')
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
        'network_architecture': 'FcNet',
        'num_hidden_units': fcnet_cfg.get('num_hidden_units', (40, 40)),
        'activation': fcnet_cfg.get('activation', 'relu'),
        'dropout_rate': fcnet_cfg.get('dropout_rate', 0.0),
        'use_layernorm': fcnet_cfg.get('use_layernorm', False),
        'train_val_split_function': tabular_train_val_split,
        'num_inner_updates': eval_cfg.get('num_inner_updates', 5),
        'inner_lr': eval_cfg.get('inner_lr', 0.01),
        'meta_lr': 1e-3,
        'KL_weight': eval_cfg.get('KL_weight', 1e-6),
        'num_models': eval_cfg.get('num_models', 4),
        'svgd_bandwidth_scale': eval_cfg.get('svgd_bandwidth_scale', 1.0),
        'svgd_repulsive_weight': eval_cfg.get('svgd_repulsive_weight', 1.0),
        'first_order': eval_cfg.get('first_order', True),
        'train_flag': False,
        'resume_epoch': best_epoch,
        'num_epochs': best_epoch,
        'num_episodes': 1,
        'num_episodes_per_epoch': 1,
        'minibatch': 1,
        'minibatch_print': 1,
        'logdir': log_dir,
        'classification_threshold': eval_cfg.get('classification_threshold', 0.5),
        'loss_function': FocalLoss(alpha=loss_cfg.get('alpha', 0.25), gamma=loss_cfg.get('gamma', 2.0))
    }

    bmaml = Bmaml(config=config)

    # 1. Evaluate on all standard test cases
    for tc_file in test_case_files:
        tc_name = os.path.basename(tc_file).replace('.pt', '')
        print(f"\n==========================================")
        print(f"Evaluating Fast Adaptation on: {tc_name}")
        print(f"==========================================")

        tasks = torch.load(tc_file, weights_only=False)
        model = bmaml.load_model(resume_epoch=best_epoch, eps_dataloader=None)

        for task_id, task_data in enumerate(tasks):
            x_s = task_data['support_x'].to(device)
            y_s = task_data['support_y'].to(device)
            x_q = task_data['query_x'].to(device)
            y_q = task_data['query_y'].to(device)

            ep = [{
                'x_s': x_s.unsqueeze(0),
                'y_s': y_s.unsqueeze(0),
                'x_q': x_q.unsqueeze(0),
                'y_q': y_q.unsqueeze(0),
                'task_id': task_id
            }]

            split_data = tabular_train_val_split(ep[0])
            x_t = split_data['x_t'].to(device)
            y_t = split_data['y_t'].to(device)
            x_v = split_data['x_v'].to(device)
            y_v = split_data['y_v'].to(device)

            adapted_hyper_net = bmaml.adaptation(x=x_t, y=y_t, model=model)
            logits_list = bmaml.prediction(x=x_v, adapted_hyper_net=adapted_hyper_net, model=model)
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
                model="BMAML",
                architecture="FcNet",
                mode="Fast Adaptation",
                query_size=list(x_q.shape)
            )


if __name__ == '__main__':
    main()
