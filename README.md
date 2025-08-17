# Continual Learning for Point Clouds
**LoRA + Model Merging (Simple / Fisher / TIES) + Bayesian LoRA (Laplace) + Uncertainty & Calibration**

A reproducible research scaffold to study **class‑incremental learning** on 3D point clouds.  
It trains **LoRA adapters** task‑wise on a frozen backbone, **merges** the task adapters into a single model, and evaluates **accuracy**, **calibration (ECE)**, and **uncertainty** (Entropy of the Mean, incl. OOD on ModelNet40).  
Runs out‑of‑the-box with a **PointNet baseline** and provides insertion points to drop in a **Point Transformer** backbone for exact replication of thesis‑style results.

---

## ✨ Features

- **Dataset**: ModelNet10 (ID) with optional OOD from ModelNet40  
- **Incremental setup**: 5 tasks × 2 classes each (modifiable in YAML)
- **LoRA training**: Base **frozen**; train **only LoRA (A/B)** per task
- **Merging methods**:
  - `simple`: element‑wise average
  - `fisher`: element‑wise Fisher‑weighted average
  - `ties`: **TIES‑Merging** (trim → sign election → disjoint merge)
  - `bayesian`: **Bayesian LoRA** via diagonal **Laplace**; precision‑weighted merging over S samples
- **Uncertainty & Calibration**:
  - **ECE** (default 15 bins)
  - **Entropy of the Mean (EoM)** for ID & OOD
- **Results**: Per‑task / per‑class CSVs, overall metrics JSON, optional confusion matrix plot
- **Quality of life**: Reproducible seeds, AMP, grad‑clip, progress logs

---

```

---

## 🛠️ Setup

```bash
# 1) Environment
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# 2) Data: download & preprocess OFF → NPZ (1024 points)
python data/scripts/download_modelnet.py --root data/modelnet --datasets ModelNet10 ModelNet40
python data/scripts/preprocess_modelnet.py --root data/modelnet --dataset ModelNet10 --num-points 1024
python data/scripts/preprocess_modelnet.py --root data/modelnet --dataset ModelNet40 --num-points 1024
```

> The preprocessing stores `points (N,3)` and a string `label` in compressed `.npz` under  
> `data/modelnet/ModelNetXX/{train,test}/`.

---

## 🚦 Run the pipeline

### 1) Train LoRA adapters per task
```bash
python scripts/train.py --cfg config/experiment_config.yaml --amp --grad-clip 1.0
```
Outputs per task (under `results/adapters/`):
- `taskXX_lora.pth` – LoRA (A/B) only
- `taskXX_fisher.pth` – diagonal Fisher (optional; controlled by flags)
- `taskXX_var.pth` – diagonal Laplace variances Σ ≈ (F + λI)⁻¹ (optional)

### 2) Merge adapters
```bash
# method ∈ {simple, fisher, ties, bayesian}
python scripts/merge_model.py --cfg config/experiment_config.yaml --method ties
```
Saves `results/merged_<method>.pth`.

### 3) Evaluate
```bash
# Deterministic: evaluate the merged adapter
python scripts/evaluate.py --cfg config/experiment_config.yaml
# or specify which merged adapter to use:
python scripts/evaluate.py --cfg config/experiment_config.yaml --merged results/merged_ties.pth

# Bayesian ensemble: resample S merged adapters from μ/σ² and report EoM
python scripts/evaluate.py --cfg config/experiment_config.yaml --method bayesian --samples 10
```

**Artifacts**
- `results/accuracy_overall.json` – overall accuracy, ECE, OOD entropy
- `results/accuracy_per_task.csv` – per‑task accuracy
- `results/accuracy_per_class.csv` – per‑class accuracy
- `results/calibration_ood.csv` – ECE (ID), mean OOD entropy
- `results/confusion_matrix.png` – optional plot

---

## ⚙️ Configuration

```yaml
# config/experiment_config.yaml
project_name: cl_pointcloud
seed: 42

data:
  root: data/modelnet
  dataset: ModelNet10
  num_points: 1024
  memory_num_points: 100                 # for optional memory FT
  tasks:
    - [chair, sofa]
    - [bed, bathtub]
    - [desk, dresser]
    - [monitor, table]
    - [night_stand, toilet]
  ood_dataset: ModelNet40
  ood_num_classes: 10
  ood_seed: 123

training:
  device: cuda
  epochs_per_task: 20
  batch_size: 32
  lr: 0.001
  weight_decay: 0.0
  optimizer: adam

model:
  backbone: point_transformer            # fallback to PointNet baseline
  num_classes: 10
  lora:
    rank: 8
    alpha: 16
    dropout: 0.0

merging:
  method: ties
  ties:
    trim_percentile: 0.2

uncertainty:
  laplace:
    approx: diagonal
    samples: 10

evaluation:
  ece_bins: 15
  save_dir: results
```

> 🔁 Change the `tasks` list to alter the class‑incremental schedule.  
> 🧪 For quick tests, reduce `epochs_per_task`, `num_points`, or batch size.

---

## 🔌 Backbones

- **PointNet baseline** (`models/pointnet_baseline.py`) ships ready to run—great for smoke tests.
- **Point Transformer** (`models/point_transformer.py`) is a **stub**.  
  Plug in your implementation (keep the classifier head API) and LoRA will automatically wrap `nn.Linear` layers.

---

## 🧩 LoRA, Fisher & Laplace (How it fits)

- **LoRA injection**: during training, we freeze the base network and add low‑rank adapters to all `nn.Linear` layers. Only `lora_A`/`lora_B` train per task.
- **Fisher diagonal**: estimated over a few batches for LoRA params; used for Fisher‑weighted merging.
- **Laplace variances**: Σ ≈ (F + λI)⁻¹ (diagonal); used to **sample** per‑task adapters and **precision‑weight** them in Bayesian merging/evaluation.
- **TIES‑Merging**: removes small‑magnitude noise (trim), chooses a majority sign per parameter (elect), and merges only sign‑consistent updates (disjoint).

---

## 📊 Reproducing typical tables

- **Incremental accuracy (per task + final)** → `results/accuracy_per_task.csv`  
- **Calibration (ECE) & OOD uncertainty** → `results/calibration_ood.csv`  
- **Confusion matrix & per‑class stats** → `results/confusion_matrix.png`, `results/accuracy_per_class.csv`

> Exact numbers depend on the **backbone** (Point Transformer vs PointNet), data sampling, and random seeds.  
> Use the Point Transformer for a fair comparison with Point‑Transformer‑based results.

---

## 🧪 Tips & Troubleshooting

- **Slow preprocessing?** Use parallel workers:  
  `python data/scripts/preprocess_modelnet.py --workers 8 ...`
- **OOM / speed**:
  - Reduce `batch_size` or `num_points`
  - Use `--amp` in training for mixed precision (GPU)
- **Fisher or Laplace missing?** Re‑run `train.py` without `--no-fisher/--no-laplace`, or compute Fisher only once with small `--fisher-max-batches`.
- **Merged file not found in `evaluate.py`?** Pass `--merged results/merged_ties.pth`.

---

