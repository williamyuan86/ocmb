# OCMB: Ordering-Constrained Markov Blanket Discovery

[![Paper](https://img.shields.io/badge/Paper-ICML%202026-blue)](https://icml.cc/virtual/2026/poster/65568)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **"Global Directional Priors with Local Statistical Validation for Scalable Causal Discovery"** (ICML 2026).

**Paper**: [https://icml.cc/virtual/2026/poster/65568](https://icml.cc/virtual/2026/poster/65568)

## Introduction

OCMB (Ordering-Constrained Markov Blanket) is a two-stage causal discovery algorithm:

1. **Stage 1**: Use an ordering backbone (e.g., SciNO or CaPS) to obtain topological ordering and parent scores
2. **Stage 2**: Learn Markov Blankets using constrained IAMB under ordering constraints, then orient edges based on the topological order

### Key Advantages

- **Modular Design**: Supports multiple ordering backbones (Oracle, Random, SciNO, CaPS, etc.)
- **Candidate Set Constraints**: Constructs candidate parent sets using topological ordering and parent scores, drastically reducing search space
- **Scalability**: Suitable for high-dimensional data (supports GPU-accelerated CMI tests)
- **Theoretical Guarantees**: Guarantees correct Markov Blanket discovery when candidate sets cover true parents

## Directory Structure

```
OCMB/
├── __init__.py              # Package initialization
├── ocmb.py                  # OCMB core implementation
├── caps/                    # CaPS backbone
│   ├── __init__.py
│   ├── caps_backend.py      # CaPS main algorithm
│   └── utils.py             # CaPS utility functions
├── scino/                   # SciNO backbone
│   └── sciNO/               # SciNO module
│       ├── __init__.py
│       ├── scino.py         # SciNO main algorithm
│       ├── models/          # Neural network models
│       ├── trainers/        # Training modules
│       └── data/            # Data loaders
└── utils/                   # Utility modules
    ├── __init__.py
    ├── KnnCMI_loader.py     # CMI test unified loader
    ├── KnnCMI_cuda.py       # GPU-accelerated CMI
    ├── KnnCMI_cuda_cached.py # Cached GPU CMI
    └── KnnCMI.py            # CPU CMI
```

## Installation

```bash
pip install numpy pandas scipy scikit-learn networkx torch
pip install numba  # For GPU-accelerated CMI (optional)
pip install lightgbm  # For CaPS_LGBM variant (optional)
```

## Quick Start

### Simple Example

See `example.py` for a quick introduction:

```bash
python example.py
```

### Reproducing Paper Results

The `examples/` directory contains scripts to reproduce key experiments from the paper:

```bash
cd examples/

# 1. Main synthetic graph experiments (Table 2)
python 1_synthetic_graphs.py

# 2. K sensitivity analysis (Table 9, Figure 8)
python 2_k_sensitivity.py

# 3. Runtime comparison (Figure 7)
python 3_runtime_comparison.py

# 4. Ordering robustness (Figure 10)
python 4_ordering_robustness.py
```

See `examples/README.md` for detailed documentation and expected results.

### Basic Usage

```python
import numpy as np
from OCMB import run_ocmb, calculate_metrics

# Prepare data
X = np.random.randn(500, 10)  # (n_samples, n_nodes)

# Run OCMB with CaPS backbone
graph, ocmb = run_ocmb(
    X,
    backbone='caps',           # Choose backbone: 'caps', 'scino', 'random', 'oracle'
    max_parents=10,            # Maximum candidate parent set size
    k_mb=5,                    # k-NN parameter for CMI tests
    alpha_mb=0.01,             # CMI significance threshold
    verbose=True
)

# Get results
print(f"Number of edges: {int(np.sum(graph))}")
print(f"Topological order: {ocmb.order_}")
print(f"CMI calls: {ocmb.get_n_cmi_calls()}")
```

### Using Class Interface

```python
from OCMB import OCMB_CaPS

# Create OCMB instance
ocmb = OCMB_CaPS(
    max_parents=10,
    k_mb=5,
    alpha_mb=0.01,
    eta_G=0.001,          # CaPS parameter
    eta_H=0.001,
    device='cuda:0',
    verbose=True
)

# Fit data
ocmb.fit(X, true_adj=true_graph)  # true_adj for computing ordering divergence

# Get adjacency matrix
graph = ocmb.get_adjacency_matrix()

# Get timing information
timings = ocmb.get_timings()
print(f"Ordering stage: {timings['ordering']:.2f}s")
print(f"MB learning stage: {timings['mb']:.2f}s")
print(f"Total time: {timings['total']:.2f}s")
```

## Available Backbones

### 1. CaPS (Recommended for continuous data)

```python
from OCMB import OCMB_CaPS

ocmb = OCMB_CaPS(
    eta_G=0.001,          # Stein gradient regularization
    eta_H=0.001,          # Stein Hessian regularization
    dispersion='mean',
    device='cuda:0',
    max_parents=10,
    k_mb=5,
    alpha_mb=0.01
)
```

### 2. CaPS + LightGBM Score (Sharper parent scoring)

```python
from OCMB import OCMB_CaPS_LGBMScore

ocmb = OCMB_CaPS_LGBMScore(
    eta_G=0.001,
    eta_H=0.001,
    lgbm_n_estimators=100,
    lgbm_fast_mode=True,  # Fast mode
    use_sample_splitting=True,  # Sample splitting
    max_parents=10,
    k_mb=5,
    alpha_mb=0.01
)
```

### 3. SciNO (Recommended for high-dimensional data)

```python
from OCMB import OCMB_SciNO

ocmb = OCMB_SciNO(
    score_hidden=256,
    score_epochs=100,
    op_width=128,
    op_epochs=60,
    masking=True,
    device='cuda:0',
    max_parents=10,
    k_mb=5,
    alpha_mb=0.01
)
```

### 4. Oracle (Theoretical upper bound, requires true DAG)

```python
from OCMB import OCMB_Oracle

ocmb = OCMB_Oracle(
    max_parents=10,
    k_mb=5,
    alpha_mb=0.01
)
ocmb.fit(X, true_adj=true_graph)  # Must provide true_adj
```

### 5. Random (Ablation baseline)

```python
from OCMB import OCMB_Random

ocmb = OCMB_Random(
    seed=42,
    max_parents=10,
    k_mb=5,
    alpha_mb=0.01
)
```

## Key Parameters

### OCMB Common Parameters

- `max_parents` (int): Maximum candidate parent set size K
  - Suggested: K=5-10 for low-dim, K=0.2d-0.4d for high-dim
- `k_mb` (int): k-NN parameter for Markov Blanket learning
  - Suggested: 3-10, use larger values for large sample sizes
- `alpha_mb` (float): CMI significance threshold
  - Suggested: 0.01-0.05, smaller values are more conservative
- `symmetry` (str): MB symmetry strategy
  - 'delete': Remove asymmetric edges (conservative, recommended)
  - 'add': Add asymmetric edges (aggressive)
- `use_spouse_closure` (bool): Include spouse closure in CandNei
  - Recommended: True (default)
- `score_threshold` (float): Parent score threshold τ
  - Filters low-score candidates to prevent random baseline inflation
- `score_threshold_quantile` (float): Auto-compute τ by quantile
  - e.g., 0.8 keeps only top 20% high-score candidates

### CaPS Parameters

- `eta_G` (float): Regularization for Stein gradient estimation (default 0.001)
- `eta_H` (float): Regularization for Stein Hessian estimation (default 0.001)
- `device` (str): Computing device ('cuda:0' or 'cpu')

### SciNO Parameters

- `score_epochs` (int): Training epochs for score network (default 100)
- `op_epochs` (int): Training epochs for FNO operator (default 60)
- `score_hidden` (int): Hidden dimension for score network (default 256)
- `op_width` (int): Width of FNO operator (default 128)
- `high_dim_threshold` (int): High-dimensional threshold (default 100)
  - Automatically uses lightweight configuration above this dimension

## Evaluation Metrics

```python
from OCMB import calculate_metrics

# Compute graph evaluation metrics
metrics = calculate_metrics(true_graph, pred_graph)

print(f"SHD (Structural Hamming Distance): {metrics['SHD']}")
print(f"Precision: {metrics['Precision']:.3f}")
print(f"Recall: {metrics['Recall']:.3f}")
print(f"F1: {metrics['F1']:.3f}")
print(f"AUPR: {metrics['AUPR']:.3f}")
```

## Advanced Usage

### Precomputed Ordering Reuse

For parameter sweep experiments, precompute ordering and parent_scores, then reuse:

```python
from OCMB import OCMB_CaPS, OCMB_Precomputed

# Step 1: Get ordering using CaPS
ocmb_caps = OCMB_CaPS(verbose=False)
ocmb_caps.fit(X)
order = ocmb_caps.order_
parent_scores = ocmb_caps.parent_scores_

# Step 2: Reuse ordering in K-sweep
for K in [5, 10, 15, 20]:
    ocmb = OCMB_Precomputed(
        order=order,
        parent_scores=parent_scores,
        max_parents=K,
        k_mb=5,
        alpha_mb=0.01
    )
    ocmb.fit(X)
    # ... evaluate results
```

### Candidate Coverage Diagnostics

```python
ocmb.fit(X, true_adj=true_graph)

# Get candidate coverage statistics
print(f"Parent coverage: {ocmb.cand_stats_['covPa_mean']:.3f}")
print(f"MB coverage: {ocmb.cand_stats_['covMB_mean']:.3f}")
print(f"Avg candidate parent set size: {ocmb.cand_stats_['avg_CandPa_size']:.1f}")
```

## Performance Optimization Tips

1. **GPU Acceleration**: Use GPU device for CaPS/SciNO computation and CMI tests
2. **Sample Splitting**: Enable `use_sample_splitting=True` for large datasets
3. **Lightweight SciNO**: Automatically enabled for high-dim, or manually set `use_light_operator=True`
4. **Cache CMI**: Set environment variable `KNNCMI_USE_CACHE=1` for debugging

## FAQ

### Q1: How to choose a backbone?

- **Continuous data, low-to-medium dim (<100 nodes)**: CaPS
- **High-dimensional data (>100 nodes)**: SciNO
- **Need sharpest parent scoring**: CaPS_LGBMScore

### Q2: How to choose max_parents?

- If true max in-degree d_max is known, set K = 1.5 * d_max
- Otherwise, use K=5-10 for low-dim, K=0.2d-0.4d for high-dim
- Tune using candidate coverage diagnostics

### Q3: Why does Random baseline perform well?

- Check if `score_threshold` or `score_threshold_quantile` is set
- Random with uniform scores makes τ threshold ineffective, leading to oversized candidate sets
- Solution: Code is now fixed, Random uses truly random scores

### Q4: How to speed up?

- Use GPU: `device='cuda:0'`
- Reduce epochs: `score_epochs=60, op_epochs=40`
- Enable fast mode: `lgbm_fast_mode=True`
- Use precomputation: Run backbone once, reuse ordering

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{ocmb2026,
  title={Global Directional Priors with Local Statistical Validation for Scalable Causal Discovery},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026},
  url={https://icml.cc/virtual/2026/poster/65568}
}
```

## Paper and Resources

- **Paper**: [ICML 2026 Poster](https://icml.cc/virtual/2026/poster/65568)
- **GitHub**: [https://github.com/williamyuan86/ocmb](https://github.com/williamyuan86/ocmb)
- **Examples**: See `examples/` directory for reproducible experiments

## License

MIT License

## Contact

For issues, please open an issue or contact the developers.
