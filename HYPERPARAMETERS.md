# Tabular Few-Shot Meta-Learning: Comprehensive Hyperparameter Reference

This document provides a complete reference for every hyperparameter in the tabular few-shot meta-learning pipeline, spanning data preprocessing, neural architecture definitions, meta-training, fast adaptation, cold-start inference, and algorithm-specific parameters across **MAML**, **PLATIPUS**, **VAMPIRE**, **ABML**, and **BMAML**.

---

## Table of Contents
1. [Centralized Configuration (`tabular/config.json`)](#1-centralized-configuration-tabularconfigjson)
2. [Dataset & Preprocessing Hyperparameters (Stage 1)](#2-dataset--preprocessing-hyperparameters-stage-1)
3. [Base Network (`FcNet`) Architecture Hyperparameters](#3-base-network-fcnet-architecture-hyperparameters)
4. [Meta-Training Hyperparameters (Stage 2)](#4-meta-training-hyperparameters-stage-2)
5. [Fast Adaptation Evaluation Hyperparameters (Stage 3)](#5-fast-adaptation-evaluation-hyperparameters-stage-3)
6. [Cold-Start / Zero-Shot Evaluation Hyperparameters (Stage 4)](#6-cold-start--zero-shot-evaluation-hyperparameters-stage-4)
7. [Focal Loss Hyperparameters](#7-focal-loss-hyperparameters)
8. [Algorithm-Specific Hyperparameters (MAML, VAMPIRE, PLATIPUS, ABML, BMAML)](#8-algorithm-specific-hyperparameters)
9. [Algorithm Hyperparameter Applicability Matrix](#9-algorithm-hyperparameter-applicability-matrix)
10. [Hyperparameter Tuning Guide & Best Practices](#10-hyperparameter-tuning-guide--best-practices)

---

## 1. Centralized Configuration (`tabular/config.json`)

The entire pipeline is governed by `tabular/config.json`. Below is the complete schema:

```json
{
  "constant": {
    "steps_per_day": 144,
    "note": "1 day = 144 time steps (24 hours at 10-minute intervals). Ground truth constant."
  },
  "preprocessing": {
    "raw_dir": "tabular/data/raw",
    "output_dir": "tabular/data/processed",
    "primary_csv": "dataset_prepro_routine_generated.csv",
    "seed": 42,
    "train_frac": 0.85,
    "val_frac": 0.15,
    "test_frac": 0,
    "support_days": 1,
    "tsls_cap": 1220.0
  },
  "fcnet": {
    "num_hidden_units": [40, 40],
    "activation": "relu",
    "dropout_rate": 0.0,
    "use_layernorm": false
  },
  "meta_training": {
    "device": "cuda",
    "num_epochs": 20,
    "num_episodes_per_epoch": 1000,
    "minibatch": 5,
    "minibatch_print": 250,
    "meta_lr": 0.001,
    "inner_lr": 0.01,
    "num_inner_updates": 5,
    "first_order": true,
    "num_models": 4,
    "KL_weight": 1e-06,
    "classification_threshold": 0.5,
    "svgd_bandwidth_scale": 1.0,
    "svgd_repulsive_weight": 1.0,
    "gamma_prior_concentration": 1.0,
    "gamma_prior_rate": 0.01,
    "normal_prior_loc": 0.0,
    "normal_prior_scale": 1.0,
    "focal_loss": {
      "alpha": 0.25,
      "gamma": 2.0
    }
  },
  "fast_adaptation_eval": {
    "device": "cuda",
    "support_days": 1,
    "inner_lr": 0.01,
    "num_inner_updates": 5,
    "first_order": true,
    "num_models": 4,
    "KL_weight": 1e-06,
    "classification_threshold": 0.5,
    "svgd_bandwidth_scale": 1.0,
    "svgd_repulsive_weight": 1.0,
    "gamma_prior_concentration": 1.0,
    "gamma_prior_rate": 0.01,
    "normal_prior_loc": 0.0,
    "normal_prior_scale": 1.0,
    "focal_loss": {
      "alpha": 0.25,
      "gamma": 2.0
    }
  },
  "cold_start_eval": {
    "device": "cuda",
    "support_days": 1,
    "num_models": 4,
    "classification_threshold": 0.5,
    "gamma_prior_concentration": 1.0,
    "gamma_prior_rate": 0.01,
    "normal_prior_loc": 0.0,
    "normal_prior_scale": 1.0,
    "focal_loss": {
      "alpha": 0.25,
      "gamma": 2.0
    }
  }
}
```

---

## 2. Dataset & Preprocessing Hyperparameters (Stage 1)

These parameters control how continuous tabular time-series data (10-minute intervals, 144 steps/day) are converted into episodic tasks.

| Parameter | Type | Default | Description & Mathematical Role |
| :--- | :---: | :---: | :--- |
| **`support_days`** | `int` | `1` | Number of continuous days in the support set $\mathcal{D}_s$. With 10-minute intervals, $K = \text{support\_days} \times 144$ (e.g., $1 \text{ day} = 144$ time-steps). |
| **`query_days`** | `int` | `2` | Number of continuous days in the query set $\mathcal{D}_q$. Evaluates how well the adapted model predicts the future ($2 \text{ days} = 288$ time-steps). |
| **`step_stride`** | `int` | `144` | Sliding window step size in time-steps when generating consecutive episodes from a subject's longitudinal timeline. `144` means shifting by 1 full day. |
| **`train_frac`** | `float` | `0.85` | Proportion of generated episodic tasks allocated to `dataset_prepro_routine_generated_train_tasks.pt`. |
| **`val_frac`** | `float` | `0.15` | Proportion of generated episodic tasks allocated to `dataset_prepro_routine_generated_val_tasks.pt`. Used for validation loss & checkpoint selection. |
| **`test_frac`** | `float` | `0.0` | Proportion of generated episodic tasks allocated to `dataset_prepro_routine_generated_test_tasks.pt`. |
| **`tsls_cap`** | `float` | `1220.0` | Upper outlier threshold (in minutes) for Time Since Last Smoke (`TSLS`). Clips extreme values before StandardScaler normalization. |
| **`seed`** | `int` | `42` | Random seed for deterministic train/val/test splits and reproducible dataset preprocessing. |

### Global Categorical Vocabularies
* **`LOCATION_CLASSES` (11 Categories)**: `'Home'`, `'Public Place'`, `'Office'`, `'Construction Site'`, `'Factory'`, `'Restaurant'`, `'Gym'`, `'Entertainment'`, `'Market'`, `'School'`, `'Hospital'`.
* **`EMP_STAT_CLASSES` (12 Categories)**: `10`, `11`, `12`, `13`, `14`, `101`, `102`, `103`, `104`, `105`, `106`, `108`.

---

## 3. Base Network (`FcNet`) Architecture Hyperparameters

Configures the underlying feedforward neural network backbone defined in `CommonModels.py`.

| Parameter | Type | Default | Options | Description & Impact |
| :--- | :---: | :---: | :---: | :--- |
| **`num_hidden_units`** | `list[int]` | `[40, 40]` | `[40, 40]`, `[64, 64, 32]`, `[128, 64]`, `[32]` | Hidden layer dimensions. Controls the depth and capacity of the neural network. (e.g., `[40, 40]` yields ~2,842 parameters; `[64, 64, 32]` yields ~8,418 parameters). |
| **`activation`** | `str` | `"relu"` | `"relu"`, `"leaky_relu"`, `"elu"`, `"gelu"`, `"tanh"` | Non-linear activation function between linear layers. `"leaky_relu"` prevents dying ReLU neurons on sparse tabular features; `"gelu"` provides smooth probabilistic gating. |
| **`dropout_rate`** | `float` | `0.0` | `0.0` to `0.5` | Dropout probability applied after each hidden layer. `0.0` disables dropout; `0.05`–`0.2` adds regularization against few-shot memorization. |
| **`use_layernorm`** | `bool` | `false` | `true`, `false` | When `true`, inserts `nn.LayerNorm(dim)` after each linear projection before the activation function. Stabilizes inner-loop gradient magnitudes across heterogeneous tasks. |
| **`num_ways`** | `int` | `2` | `2` | Output dimension / number of classification classes (0 = Non-Smoking, 1 = Smoking). |

---

## 4. Meta-Training Hyperparameters (Stage 2)

These parameters control the bilevel outer/inner optimization during meta-training.

| Parameter | Type | Default | Recommended Range | Description & Mathematical Role |
| :--- | :---: | :---: | :---: | :--- |
| **`device`** | `str` | `"cuda"` | `"cuda"`, `"cpu"` | Hardware device for PyTorch tensor computation and gradient execution. |
| **`num_epochs`** | `int` | `20` | `5` – `50` | Total outer-loop training epochs. One epoch corresponds to `num_episodes_per_epoch` task episodes. |
| **`num_episodes_per_epoch`** | `int` | `1000` | `200` – `2000` | Number of task episodes sampled per epoch. Model checkpoints are evaluated and saved at the end of each epoch. |
| **`minibatch`** | `int` | `5` | `1` – `20` | Outer-loop meta-batch size $B$. Meta-gradients are accumulated across $B$ independent task episodes before updating global parameters: $\theta \leftarrow \theta - \gamma \frac{1}{B} \sum_{i=1}^B \nabla_\theta \mathcal{L}_{\text{val}}^{(i)}$. |
| **`minibatch_print`** | `int` | `250` | `50` – `500` | Interval of tasks processed before printing average training loss and accuracy to the console. |
| **`meta_lr`** | `float` | `1e-3` | `1e-4` – `5e-3` | Outer-loop learning rate $\gamma$ for the Adam optimizer updating global meta-parameters. |
| **`inner_lr`** | `float` | `0.01` | `0.001` – `0.1` | Inner-loop task adaptation step size $\alpha$. Used for task gradient descent: $\theta' = \theta - \alpha \nabla_\theta \mathcal{L}_{\text{train}}(x_s, y_s)$. |
| **`num_inner_updates`** | `int` | `5` | `1` – `10` | Number of gradient adaptation steps $K$ performed on the support set during task adaptation. |
| **`first_order`** | `bool` | `true` | `true`, `false` | When `true`, uses First-Order MAML (FOMAML), ignoring second-order Hessian-vector products ($\nabla_\theta \nabla_\theta$) for ~3x faster training and reduced GPU VRAM consumption. |
| **`num_models`** | `int` | `4` | `2` – `16` | Number of Monte Carlo weight samples $M$ (for VAMPIRE, PLATIPUS, ABML) or particles (for BMAML) drawn to approximate expectations: $\mathbb{E}_{q(\theta)}[\mathcal{L}(f_\theta(x), y)] \approx \frac{1}{M}\sum_{m=1}^M \mathcal{L}(f_{\theta^{(m)}}(x), y)$. |
| **`KL_weight`** | `float` | `1e-6` | `1e-8` – `1e-3` | Multiplier $\beta$ scaling the Kullback-Leibler (KL) divergence penalty $D_{\text{KL}}(q \parallel p)$ relative to the cross-entropy / focal loss in Bayesian algorithms (VAMPIRE, PLATIPUS, ABML). *(Note: BMAML uses SVGD instead of KL)*. |
| **`classification_threshold`** | `float` | `0.50` | `0.10` – `0.90` | Probability decision boundary threshold $\tau$ for assigning class 1 (Smoking): $\hat{y} = 1 \iff P(y=1 \mid x) \ge \tau$. Lower values increase recall for the minority class. |

---

## 5. Fast Adaptation Evaluation Hyperparameters (Stage 3)

These parameters control task-specific fine-tuning on unseen test cases (e.g., TestCase 1, TestCase 2).

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| **`checkpoint_mode`** | `str` | `"best"` | Mode for selecting the trained model checkpoint: `"best"` (automatically selects the epoch with the highest validation accuracy) or `"last"` (loads the final training epoch checkpoint). |
| **`support_days`** | `int` | `1` | Number of days provided in the test case support set for adaptation fine-tuning. |
| **`inner_lr`** | `float` | `0.01` | Adaptation step size $\alpha$ applied during test-time inference. |
| **`num_inner_updates`** | `int` | `5` | Number of gradient updates $K$ applied to the support set before evaluating on the query set. |
| **`first_order`** | `bool` | `true` | Gradient mode during test-time adaptation. |
| **`num_models`** | `int` | `4` | Number of Monte Carlo ensemble models evaluated on the query set. Predictions are averaged: $\bar{P}(y=1 \mid x) = \frac{1}{M}\sum_{m=1}^M P(y=1 \mid x, \theta^{(m)})$. |
| **`KL_weight`** | `float` | `1e-6` | Regularization weight applied during Bayesian inner-loop adaptation. |
| **`classification_threshold`** | `float` | `0.50` | Decision threshold $\tau$ used to compute Confusion Matrices, Precision, Recall, Accuracy, and Macro F1-Score on test query sets. |

---

## 6. Cold-Start / Zero-Shot Evaluation Hyperparameters (Stage 4)

Evaluates the raw meta-prior distribution without performing any support set adaptation ($K = 0$).

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| **`checkpoint_mode`** | `str` | `"best"` | Mode for selecting the meta-learned prior checkpoint: `"best"` (highest validation accuracy epoch) or `"last"` (final training epoch). |
| **`num_inner_updates`** | `int` | `0` | Strictly `0`. Bypasses support set fine-tuning to evaluate zero-shot generalization of the global prior. |
| **`num_models`** | `int` | `4` | Number of models sampled directly from the global prior $p(\theta)$ for ensemble prediction. |
| **`classification_threshold`** | `float` | `0.50` | Decision threshold $\tau$ for evaluating cold-start query performance. |

---

## 7. Focal Loss Hyperparameters

To combat severe class imbalance (where Non-Smoking class 0 heavily outnumbers Smoking class 1), Focal Loss modifies standard Cross-Entropy:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

| Parameter | Key in Config | Type | Default | Recommended Range | Description & Mathematical Role |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Class Weighting** | **`alpha`** | `float` | `0.25` | `0.1` – `0.75` | Weight $\alpha$ assigned to the positive class (Smoking, 1), with $(1 - \alpha)$ assigned to the negative class (0). Addresses foreground/background frequency imbalance. |
| **Focusing Parameter** | **`gamma`** | `float` | `2.0` | `0.5` – `5.0` | Modulating exponent $\gamma \ge 0$. Dynamically scales down the loss contribution from well-classified easy examples ($(1 - p_t)^\gamma \to 0$), forcing the network to focus on ambiguous and hard-to-classify smoking events. |

---

## 8. Algorithm-Specific Hyperparameters

### A. MAML (Model-Agnostic Meta-Learning)
* **File**: [`Maml.py`](file:///c:/Users/kunke/OneDrive/Desktop/BayesianMetalearning/few_shot_meta_learning/Maml.py)
* **Weight Representation**: Deterministic point estimate $\theta$.
* **Key Hyperparameters**:
  * `inner_lr` ($\alpha$): Step size for task adaptation: $\theta' = \theta - \alpha \nabla_\theta \mathcal{L}(x_s, y_s)$.
  * `num_inner_updates` ($K$): Adaptation step count.
  * `first_order`: True for FOMAML, False for second-order MAML.

### B. VAMPIRE (Variational Meta-Learner with Pointwise Regularization)
* **File**: [`Vampire2.py`](file:///c:/Users/kunke/OneDrive/Desktop/BayesianMetalearning/few_shot_meta_learning/Vampire2.py)
* **Weight Representation**: Variational Gaussian distribution $\mathcal{N}(\mu, \operatorname{diag}(\sigma^2))$.
* **Key Hyperparameters**:
  * `num_models` ($M$): Number of Monte Carlo parameter samples $\theta^{(m)} = \mu + \epsilon \odot \sigma$ drawn per episode.
  * `KL_weight` ($\beta$): Scales standard Normal prior penalty $D_{\text{KL}}(q \parallel \mathcal{N}(0, I))$.

### C. PLATIPUS (Probabilistic MAML with Variational Sampling)
* **File**: [`Platipus.py`](file:///c:/Users/kunke/OneDrive/Desktop/BayesianMetalearning/few_shot_meta_learning/Platipus.py)
* **Weight Representation**: Amortized initializations $\theta^{(0)} \sim q(\theta \mid \mathcal{D}_s)$.
* **Key Hyperparameters**:
  * `gamma_p`: Meta-learned initial learning rate for adapting mean parameter $\mu_\theta$.
  * `gamma_q`: Meta-learned learning rate for adapting variational posterior $q$.
  * `KL_weight` ($\beta$): Scales outer-loop KL regularization between variational posterior and prior.

### D. ABML (Amortized Bayesian Meta-Learning)
* **File**: [`Abml.py`](file:///c:/Users/kunke/OneDrive/Desktop/BayesianMetalearning/few_shot_meta_learning/Abml.py)
* **Weight Representation**: Variational Gaussian distribution $\mathcal{N}(\mu, \operatorname{diag}(\sigma^2))$.
* **Key Hyperparameters**:
  * `KL_weight` ($\beta$): Scales task-specific ELBO KL penalty $D_{\text{KL}}(q(\theta \mid \mathcal{D}_s) \parallel p(\theta))$.
  * `gamma_prior_concentration` (default: `1.0`): Gamma prior shape hyperparameter $\alpha_0$ on precision $\tau = \exp(-2 \log \sigma)$.
  * `gamma_prior_rate` (default: `0.01`): Gamma prior rate hyperparameter $\beta_0$ on precision $\tau$.
  * `normal_prior_loc` (default: `0.0`): Prior mean $\mu_0$ on weight distribution means.
  * `normal_prior_scale` (default: `1.0`): Prior scale $\sigma_0$ on weight distribution means.

### E. BMAML (Bayesian MAML with Stein Variational Gradient Descent)
* **File**: [`Bmaml.py`](file:///c:/Users/kunke/OneDrive/Desktop/BayesianMetalearning/few_shot_meta_learning/Bmaml.py)
* **Weight Representation**: Non-parametric ensemble of $M$ particles $\{\theta^{(m)}\}_{m=1}^M$.
* **Key Hyperparameters**:
  * `num_models` ($M$): Number of particles maintained in the ensemble.
  * `svgd_bandwidth_scale` (default: `1.0`): Multiplier on the RBF kernel bandwidth $h = \text{median\_dist} / \log(M)$. Larger values smooth particle interaction; smaller values encourage localized particle separation.
  * `svgd_repulsive_weight` (default: `1.0`): Multiplier on the repulsive gradient $\nabla_{\theta_j} k(\theta_i, \theta_j)$ preventing particle collapse.
  * *(Note: BMAML does not use `KL_weight` because SVGD updates particles directly via kernelized Stein discrepancy rather than variational ELBO optimization).*

---

## 9. Algorithm Hyperparameter Applicability Matrix

| Hyperparameter | Key in `config.json` | MAML | VAMPIRE | PLATIPUS | ABML | BMAML |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Inner Learning Rate** | `inner_lr` | ✅ Active | ✅ Active | ✅ Active | ✅ Active | ✅ Active |
| **Inner Step Count** | `num_inner_updates` | ✅ Active | ✅ Active | ✅ Active | ✅ Active | ✅ Active |
| **First Order Mode** | `first_order` | ✅ Active | ✅ Active | ✅ Active | ✅ Active | ✅ Active |
| **Monte Carlo Ensemble Count** | `num_models` | ❌ Fixed (1) | ✅ Active | ✅ Active | ✅ Active | ✅ Active (Particles) |
| **KL Divergence Regularization** | `KL_weight` | ❌ N/A | ✅ Active | ✅ Active | ✅ Active | ❌ N/A (Uses SVGD) |
| **SVGD Bandwidth Scale** | `svgd_bandwidth_scale` | ❌ N/A | ❌ N/A | ❌ N/A | ❌ N/A | ✅ Active |
| **SVGD Repulsive Weight** | `svgd_repulsive_weight` | ❌ N/A | ❌ N/A | ❌ N/A | ❌ N/A | ✅ Active |
| **Gamma Prior Concentration** | `gamma_prior_concentration` | ❌ N/A | ❌ N/A | ❌ N/A | ✅ Active | ❌ N/A |
| **Gamma Prior Rate** | `gamma_prior_rate` | ❌ N/A | ❌ N/A | ❌ N/A | ✅ Active | ❌ N/A |
| **Normal Prior Scale** | `normal_prior_scale` | ❌ N/A | ❌ N/A | ❌ N/A | ✅ Active | ❌ N/A |
| **Classification Threshold** | `classification_threshold` | ✅ Active | ✅ Active | ✅ Active | ✅ Active | ✅ Active |
| **Focal Loss** ($\alpha, \gamma$) | `focal_loss` | ✅ Active | ✅ Active | ✅ Active | ✅ Active | ✅ Active |

---

## 10. Hyperparameter Tuning Guide & Best Practices

### A. Tuning for Class Imbalance (Smoking Detection)
1. **Decision Threshold ($\tau$)**:
   - If positive class recall is too low ($< 50\%$), reduce `classification_threshold` from `0.50` to `0.30` or `0.20`.
2. **Focal Loss Tuning**:
   - Increase `gamma` to `2.5` or `3.0` if the model is swamped by easy non-smoking periods.
   - Increase `alpha` to `0.50` or `0.60` to place heavier gradient emphasis on true smoking timestamps.

### B. Neural Architecture Tuning
1. **Topology Depth**:
   - Tabular features (29 inputs): `[40, 40]` is standard. If underfitting, upgrade to `[64, 64, 32]`.
2. **Activation Functions**:
   - Switch from `"relu"` to `"leaky_relu"` or `"gelu"` to prevent dead gradient channels on zero-padded sensor readings.
3. **Layer Normalization**:
   - Set `"use_layernorm": true` when training with higher `inner_lr` ($\ge 0.03$) to prevent internal activation explosions during inner adaptation loops.

### C. Meta-Learning Stability
1. **Inner Step Count (`num_inner_updates`)**:
   - Keep between `3` and `7`. Setting $K > 10$ can cause meta-gradient degradation and inner-loop overfitting on small 1-day support sets.
2. **Monte Carlo Samples (`num_models`)**:
   - `4` provides a solid balance between runtime and variance reduction. Increase to `8` or `16` for critical evaluation runs.
3. **KL Divergence Weight (`KL_weight`)**:
   - Recommended range: `1e-6` to `1e-4`. If set too high ($\ge 1e-2$), the variational posterior will fail to adapt away from the prior. If set too low ($\le 1e-8$), the model loses Bayesian uncertainty regularization.
4. **BMAML Particle Dynamics (`svgd_repulsive_weight`)**:
   - If particles collapse to the identical model, increase `svgd_repulsive_weight` to `2.0` or `3.0` or reduce `svgd_bandwidth_scale` to `0.5`.
