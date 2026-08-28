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

from tabular.dataloader.TabularDataLoader import tabular_train_val_split, load_config
from tabular.metrics_utils import compute_metrics
from _utils import FocalLoss


def main():
    parser = argparse.ArgumentParser(description="Evaluate MAML + LogisticRegression (Cold Start) on all test cases.")
    parser.add_argument("--test_case", type=str, default=None, help="Optional specific test case name/file to evaluate.")
    args = parser.parse_args()

    cfg = load_config()
    hparams = cfg.get('training_hyperparameters', {})
    tc_cfg = cfg.get('test_cases', {})
    loss_cfg = hparams.get('focal_loss', {'alpha': 0.25, 'gamma': 2.0})

    data_dir = os.path.join(REPO_ROOT, 'tabular', 'data', 'processed')
    all_task_files = sorted(glob.glob(os.path.join(data_dir, "*_tasks.pt")))
    test_case_files = [f for f in all_task_files if "dataset_prepro" not in os.path.basename(f)]

    if args.test_case:
        test_case_files = [f for f in test_case_files if args.test_case.lower() in os.path.basename(f).lower()]

    if not test_case_files:
        raise FileNotFoundError(f"No test case .pt files found in: {data_dir}. Run Preprocessing.py first.")

    print(f"Found {len(test_case_files)} test case(s) for zero-shot evaluation:")
    for f in test_case_files:
        print(f"  - {os.path.basename(f)}")

    k_shot = tc_cfg.get('k_shot', 144)

    config = {
        'num_ways': 2,
        'k_shot': k_shot,
        'device': torch.device('cpu'),
        'network_architecture': 'LogisticRegression',
        'train_val_split_function': tabular_train_val_split,
        'num_inner_updates': hparams.get('num_inner_updates', 5),
        'inner_lr': hparams.get('inner_lr', 0.01),
        'meta_lr': hparams.get('meta_lr', 1e-3),
        'KL_weight': hparams.get('KL_weight', 1e-6),
        'num_models': hparams.get('num_models', 4),
        'first_order': True,
        'train_flag': False,
        'resume_epoch': hparams.get('num_epochs', 20),
        'num_epochs': hparams.get('num_epochs', 20),
        'num_episodes': 1,
        'num_episodes_per_epoch': 1,
        'minibatch': 1,
        'minibatch_print': 1,
        'logdir': os.path.join(REPO_ROOT, 'tabular', 'outputs', 'logs_maml_lr'),
        'loss_function': FocalLoss(alpha=loss_cfg.get('alpha', 0.25), gamma=loss_cfg.get('gamma', 2.0))
    }

    print("\nInitializing MAML and loading pre-trained model (Epoch 20)...")
    from Maml import Maml
    maml = Maml(config=config)

    dummy_tasks = torch.load(test_case_files[0], weights_only=False)
    dummy_ep = [{
        "x_s": dummy_tasks[0]['x_s'].unsqueeze(0),
        "y_s": dummy_tasks[0]['y_s'].unsqueeze(0),
        "x_q": dummy_tasks[0]['x_q'].unsqueeze(0),
        "y_q": dummy_tasks[0]['y_q'].unsqueeze(0),
        "task_id": int(dummy_tasks[0].get('task_id', 1))
    }]
    model = maml.load_model(resume_epoch=config['num_epochs'], hyper_net_class=maml.hyper_net_class, eps_dataloader=dummy_ep)

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
        x_v = split_data['x_v'].to(config['device'])
        y_v = split_data['y_v'].to(config['device'])

        logits = maml.prediction(x=x_v, adapted_hyper_net=model["hyper_net"], model=model)
        y_pred = torch.softmax(input=logits, dim=1).argmax(dim=1).cpu().numpy()
        y_true = y_v.cpu().numpy()

        compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            test_case=tc_name,
            task_id=task_id,
            model="MAML",
            architecture="LogisticRegression",
            mode="Cold Start",
            query_size=list(x_q.shape)
        )

        print("\n" + "="*60)
        print(f"EVALUATION (COLD START): {tc_name} (Task ID {task_id}) [MAML LR]")
        print(f"Query Size: {list(x_q.shape)}")
        print("="*60)
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=["Not Smoking (0)", "Smoking (1)"], zero_division=0))
        print("="*60)


if __name__ == "__main__":
    main()
