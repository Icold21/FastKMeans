# 🚀 FastKMeans Suite: Scalable Prototype-based Machine Learning

**FastKMeans Suite** is an enterprise-grade, GPU-accelerated framework that bridges the gap between the accuracy of traditional Metric Learning (like kNN or SVM) and the massive scalability of Deep Learning.

By condensing infinite datasets into an intelligent, differentiable grid of topological prototypes, FastKMeans drops inference complexity from $\mathcal{O}(N \times D)$ to **$\mathcal{O}(K \times D)$** (and $\mathcal{O}(\log K)$ with FAISS), guaranteeing extreme inference speed without sacrificing precision.

---

## 📖 Table of Contents
1. [Installation](#-installation)
2. [Available Models](#-available-models)
3. [The Core Philosophy](#-the-core-philosophy)
4. [Master API & Stream Learning](#-master-api--stream-learning)
5. [Global Parameters Configuration](#-global-parameters-configuration)
6. [Model-Specific Parameters](#-model-specific-parameters)
7. [Advanced Features Deep Dive](#-advanced-features-deep-dive)
8. [Quick Start Example](#-quick-start-example)

---

## 📦 Installation

To install the framework locally in editable mode (so your code updates apply immediately):

```bash
git clone https://github.com/yourusername/FastKMeans.git
cd FastKMeans
pip install -e .
```

**Requirements:** `torch >= 2.0.0`, `numpy`, `scikit-learn`, `tqdm`.  
*(Optional for extreme scaling: `faiss-cpu` or `faiss-gpu`)*

---

## 🧩 Available Models

FastKMeans provides four highly specialized classes inheriting from `BaseFastKMeans`.

| Model Class | Task | Loss Function | Key Architectural Features |
| :--- | :--- | :--- | :--- |
| **`FastKMeansClusterer`** | Unsupervised Clustering | Feature Distance | Evaluates raw metric distances. Uses `diversity_reg` to prevent mode collapse. |
| **`FastKMeansClassifier`** | Multiclass Classification | Cross-Entropy | Supports **Negative Sampling** (Sampled Softmax) for extreme >100k class spaces. Features **LVQ Repulsion** to push boundaries away from wrong classes. |
| **`FastMultiLabelKMeansClassifier`**| Multi-Label / Tagging | Asymmetric Loss (ASL) | Builds **Independent Sub-topologies** per tag. Uses ASL to combat extreme sparsity. Includes zero-hyperparameter `gap` prediction strategy. |
| **`FastKMeansRegressor`** | Continuous / Vector Regression | Mean Squared Error (MSE) | Uses **2-Stage Stratification** (clusters Y, then spawns X prototypes inside). Supports multi-dimensional vector outputs. |

---

## 🧠 The Core Philosophy: "Hard Training, Soft Inference"

1. **Phase 1: Hard Stream Learning (EMA)**  
   Data flows in infinitely via `fit_batch(gradient_step=False)`. Samples are strictly assigned to the nearest centroid (creating perfect Voronoi cells). Centroids shift geometrically using Exponential Moving Averages (EMA).
2. **Phase 2: Differentiable Fine-tuning**  
   The topological grid is unfrozen. Using `fit_batch(gradient_step=True)`, the `Adam` optimizer physically moves the centroids via gradient descent to directly minimize the Loss function.
3. **Phase 3: Soft Inference (Smart kNN)**  
   During `.predict()`, targets of the `top_k` closest prototypes are smoothly blended using Inverse Distance Weighting (IDW).

---

## 🌊 Master API & Stream Learning

You are no longer constrained by RAM. The entire API revolves around the `fit_batch` master function, allowing you to stream billions of rows directly from a database.

```python
model = FastKMeansClassifier()

# 1. Build the Grid (Phase 1)
for batch_X, batch_y, batch_sw, batch_mask in infinite_stream:
    model.fit_batch(batch_X, batch_y, sample_weight=batch_sw, feature_mask=batch_mask, gradient_step=False)

# 2. Refine with Gradients (Phase 2)
model.find_learning_rate(X_sample, y_sample) # Auto-find optimal LR
for batch_X, batch_y, batch_sw, batch_mask in infinite_stream:
    model.fit_batch(batch_X, batch_y, sample_weight=batch_sw, feature_mask=batch_mask, gradient_step=True)
```

*(The helper functions `.fit()` and `.finetune()` are convenient wrappers around `fit_batch` for static datasets that fit in RAM).*

---

## ⚙️ Global Parameters Configuration

These parameters are available in the `__init__` of **all** models.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `distance` | `str` | `'cosine'` | Math metric. Options: `'cosine'`, `'euclidean'`, `'l1'`. |
| `dtype` | `str` | `'float32'` | Memory precision. Options: `'float32'`, `'float16'`, `'bfloat16'`. |
| `init_mode` | `str` | `'kmeans++'` | Prototype spawning logic. Options: `'kmeans++'`, `'random'`. |
| `soft_type` | `str` | `'scaled'` | **Crucial:** Routing logic for targets and gradients. Options: `'hard'`, `'mean'`, `'scaled'`, `'softmax_scaled'`. *(Note: 'hard' and 'mean' disable gradients)*. |
| `temperature`| `float`| `1.0` | Scalar dividing distances when `soft_type='softmax_scaled'`. |
| `top_k` | `int` | `5` | Number of prototypes aggregated during inference. Use `-1` for all. |
| `auto_feature_weights`| `bool` | `False` | Enables real-time ANOVA algorithmic feature importance scaling. |
| `negative_sampling` | `int/None` | `None` | Restricts loss calculation to $N$ random wrong classes (speeds up multiclass training). |
| `diversity_reg` | `float` | `0.0` | Orthogonality penalty enforcing prototype spreading (prevents mode collapse). |
| `l2_reg` | `float` | `0.0` | Standard L2 weight decay applied to centroid coordinates. |
| `use_faiss` | `bool` | `False` | Utilizes HNSW Graph indexing for $\mathcal{O}(\log K)$ inference (Requires `faiss`). |
| `use_compile` | `bool` | `False` | Utilizes PyTorch 2.0 Triton compiler `torch.compile` for speedups. |

---

## 🎯 Model-Specific Parameters

### 🟢 Classification & Clustering (`k_init`)
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `k_init` | `int/str` | `3` | Number of prototypes per class. If set to **`'auto'`**, uses *Geometric Information Dispersion* to analytically spawn the perfect amount of prototypes based on class complexity! |
| `repulsion_factor` | `float` | `0.05` | (LVQ) Push-force applied by negative samples to expel wrong prototypes. |

### 🟡 Multi-Label (`FastMultiLabelKMeansClassifier`)
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `asl_gamma_neg` | `float` | `4.0` | Asymmetric Loss decay for easy negative samples. |
| `asl_gamma_pos` | `float` | `1.0` | Asymmetric Loss decay for positive samples. |
| `asl_clip` | `float` | `0.05` | Margin under which negative samples are fully discarded from gradients. |

### 🔵 Regression (`FastKMeansRegressor`)
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `k_targets` | `int/str` | `20` | Stage 1: Number of $Y$-value stratas to bucket the dataset into. (Accepts `'auto'`). |
| `k_features` | `int/str` | `10` | Stage 2: Number of $X$-prototypes injected into each bucket. (Accepts `'auto'`). |
| `target_distance`| `str` | `'euclidean'`| Metric used to cluster the $Y$ variable during initialization. |
| `target_assignment`| `str` | `'scaled'` | How EMA target averages are generated. Options: `'hard', 'mean', 'scaled', 'softmax_scaled'`. |

---

## 🔬 Advanced Features Deep Dive

### 1. 🎭 Analytical 'Auto' K-Spawning
If you set `k_init='auto'`, the framework becomes **hyperparameter-free**. It calculates the *Geometric Information Dispersion* of the class:
* If a class is dense and identical (dispersion $\approx 0$), it spawns **$1$** prototype.
* If a class is chaotic and spread out, it scales up to $\mathcal{O}(\sqrt{N})$.

### 2. 🧮 Auto Feature Weights (ANOVA)
Setting `auto_feature_weights=True` launches Welford's algorithm inline. It compares the *within-cluster variance* of a feature to its *global variance*. 
Features carrying noise are automatically down-weighted in the distance math. During `.finetune()`, these weights become `nn.Parameter` and are tuned via Backpropagation!

### 3. 🛡️ Feature Masks (For NLP & Time-Series)
In `fit`, `fit_batch`, and `predict`, you can pass a `feature_mask` tensor (1 for active, 0 for pad/missing). 
* **The Magic:** The distance math completely ignores the masked features. During EMA, the framework allocates a `(K, D)` matrix so that prototypes accumulate "age" independently per coordinate, preventing padding tokens from pulling prototypes to absolute zero.

### 4. 🎛️ Auto Learning Rate Finder (CatBoost Style)
If you do not provide an `lr` argument to `.finetune()`, the framework runs a micro-simulation. It exponentially scales the learning rate, tests batches, mathematically calculates the steepest negative gradient of the loss curve, sets the optimal `LR`, and uses `copy.deepcopy` to perfectly restore the untainted topological grid before real training starts.

### 5. 🏷️ Zero-Hyperparameter Multi-Label "Gap" Strategy
Instead of blindly guessing threshold probabilities (e.g. `p > 0.5`), calling `predict(X, strategy='gap')` sorts the probabilities and automatically places the cutoff at the largest mathematical confidence drop.

---

## ⚡ Quick Start Example

```python
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from fast_kmeans import FastKMeansRegressor

# 1. Load Data
X_np, y_np = fetch_california_housing(return_X_y=True)
X = torch.tensor(StandardScaler().fit_transform(X_np), dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1) # Support Vector Targets!

# 2. Initialize Model
reg = FastKMeansRegressor(
    k_targets='auto',             # Analytically choose target buckets
    k_features=10,                # 10 prototypes per bucket
    distance='euclidean',
    soft_type='softmax_scaled',   # Differentiable routing required for gradients!
    diversity_reg=0.01            # Prevent prototypes from clustering together
)

# 3. Phase 1: EMA Topology construction
print("Building rigid Voronoi map...")
reg.fit(X, y, max_iters=20)

# 4. Phase 2: Differentiable Finetuning (Auto LR triggered)
print("Finetuning with Gradients...")
reg.finetune(X, y, epochs=30, early_stopping_rounds=5)

# 5. Inference
predictions = reg.predict(X)
print(predictions[:5])
```