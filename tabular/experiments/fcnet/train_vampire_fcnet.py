import sys
import os
from pathlib import Path

# Add repo root to sys.path so repo modules (Vampire2, Maml, etc.) are importable
REPO_ROOT = str(Path(__file__).resolve().parents[3])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from Vampire2 import Vampire2
from tabular.dataloader.TabularDataLoader import get_tabular_dataloader, tabular_train_val_split, load_config, resolve_device
from _utils import FocalLoss

# Resolve paths relative to repo root
DATA_DIR = os.path.join(REPO_ROOT, 'tabular', 'data', 'processed')
LOG_DIR = os.path.join(REPO_ROOT, 'tabular', 'outputs', 'logs_vampire_fcnet')


def main():
    cfg = load_config()
    train_cfg = cfg.get('meta_training', cfg.get('training_hyperparameters', {}))
    prepro_cfg = cfg.get('preprocessing', cfg.get('meta_training_split', {}))
    loss_cfg = train_cfg.get('focal_loss', {'alpha': 0.25, 'gamma': 2.0})

    STEPS_PER_DAY = 144
    support_days = prepro_cfg.get('support_days', 1)
    k_shot = support_days * STEPS_PER_DAY
    device = resolve_device(train_cfg.get('device', 'cuda'))

    print(f"Training on device: {device}")

    # Load dataloaders
    train_dataloader = get_tabular_dataloader(
        os.path.join(DATA_DIR, 'dataset_prepro_routine_generated_train_tasks.pt'), device=device)
    val_dataloader = get_tabular_dataloader(
        os.path.join(DATA_DIR, 'dataset_prepro_routine_generated_val_tasks.pt'), device=device)

    os.makedirs(LOG_DIR, exist_ok=True)

    config = {
        'num_ways': 2,
        'k_shot': k_shot,
        'device': device,

        # Architecture
        'network_architecture': 'FcNet',
        'train_val_split_function': tabular_train_val_split,

        # Meta-learning hyperparameters
        'num_inner_updates': train_cfg.get('num_inner_updates', 5),
        'inner_lr': train_cfg.get('inner_lr', 0.01),
        'meta_lr': train_cfg.get('meta_lr', 1e-3),
        'KL_weight': train_cfg.get('KL_weight', 1e-6),
        'num_models': train_cfg.get('num_models', 4),

        # Gradient config
        'first_order': train_cfg.get('first_order', True),
        'train_flag': True,

        # Epochs and logging
        'resume_epoch': 0,
        'num_epochs': train_cfg.get('num_epochs', 20),
        'num_episodes': 100,
        'num_episodes_per_epoch': train_cfg.get('num_episodes_per_epoch', 1000),
        'minibatch': train_cfg.get('minibatch', 5),
        'minibatch_print': train_cfg.get('minibatch_print', 250),
        'logdir': LOG_DIR,

        # Loss
        'loss_function': FocalLoss(alpha=loss_cfg.get('alpha', 0.25), gamma=loss_cfg.get('gamma', 2.0))
    }

    print("Initializing VAMPIRE...")
    vampire = Vampire2(config=config)

    print("Starting training...")
    vampire.train(train_dataloader=train_dataloader,
                  val_dataloader=val_dataloader)

    # Meta-test if available
    test_path = os.path.join(DATA_DIR, 'dataset_prepro_routine_generated_test_tasks.pt')
    if os.path.exists(test_path):
        test_tasks = torch.load(test_path, weights_only=False)
        if len(test_tasks) > 0:
            test_dataloader = get_tabular_dataloader(test_path, device=device)
            print("\n--- Training complete. Running meta-test evaluation ---")
            vampire.test(num_eps=min(50, len(test_tasks)), eps_dataloader=test_dataloader)
        else:
            print("\n--- Training complete. (Meta-test split was 0, skipping meta-test) ---")

    print("VAMPIRE training completed successfully!")


if __name__ == '__main__':
    main()
