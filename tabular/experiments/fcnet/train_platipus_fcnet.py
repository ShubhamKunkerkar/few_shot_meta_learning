import sys
import os
from pathlib import Path

# Add repo root to sys.path so repo modules (Vampire2, Maml, etc.) are importable
REPO_ROOT = str(Path(__file__).resolve().parents[3])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from Platipus import Platipus
from tabular.dataloader.TabularDataLoader import get_tabular_dataloader, tabular_train_val_split, load_config
from _utils import FocalLoss

# Resolve paths relative to repo root
DATA_DIR = os.path.join(REPO_ROOT, 'tabular', 'data', 'processed')
LOG_DIR = os.path.join(REPO_ROOT, 'tabular', 'outputs', 'logs_platipus_fcnet')


def main():
    cfg = load_config()
    hparams = cfg.get('training_hyperparameters', {})
    meta_split = cfg.get('meta_training_split', {})
    loss_cfg = hparams.get('focal_loss', {'alpha': 0.25, 'gamma': 2.0})

    # Load dataloaders
    train_dataloader = get_tabular_dataloader(
        os.path.join(DATA_DIR, 'dataset_prepro_routine_generated_train_tasks.pt'))
    val_dataloader = get_tabular_dataloader(
        os.path.join(DATA_DIR, 'dataset_prepro_routine_generated_val_tasks.pt'))
    test_dataloader = get_tabular_dataloader(
        os.path.join(DATA_DIR, 'dataset_prepro_routine_generated_test_tasks.pt'))

    os.makedirs(LOG_DIR, exist_ok=True)

    # Setup configuration directly from config.json
    config = {
        'num_ways': 2,
        'k_shot': meta_split.get('k_shot', 144),
        'device': torch.device('cpu'),

        # Architecture
        'network_architecture': 'FcNet',
        'train_val_split_function': tabular_train_val_split,

        # Meta-learning hyperparameters
        'num_inner_updates': hparams.get('num_inner_updates', 5),
        'inner_lr': hparams.get('inner_lr', 0.01),
        'meta_lr': hparams.get('meta_lr', 1e-3),
        'KL_weight': hparams.get('KL_weight', 1e-6),
        'num_models': hparams.get('num_models', 4),

        # Gradient config
        'first_order': hparams.get('first_order', True),
        'train_flag': True,

        # Epochs and logging
        'resume_epoch': 0,
        'num_epochs': hparams.get('num_epochs', 20),
        'num_episodes': 100,
        'num_episodes_per_epoch': hparams.get('num_episodes_per_epoch', 1000),
        'minibatch': hparams.get('minibatch', 5),
        'minibatch_print': hparams.get('minibatch_print', 250),
        'logdir': LOG_DIR,

        # Loss
        'loss_function': FocalLoss(alpha=loss_cfg.get('alpha', 0.25), gamma=loss_cfg.get('gamma', 2.0))
    }

    print("Initializing PLATIPUS...")
    platipus = Platipus(config=config)

    print("Starting training...")
    platipus.train(train_dataloader=train_dataloader,
                   val_dataloader=val_dataloader)

    print("\n--- Training complete. Running evaluation ---")
    platipus.test(num_eps=50, eps_dataloader=test_dataloader)

    print("PLATIPUS training and evaluation completed successfully!")


if __name__ == '__main__':
    main()
