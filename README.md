# 🚀 FastKMeans Suite: The Scalable Prototype-Learning Framework

**FastKMeans** bridges the gap between the accuracy of traditional Metric Learning (like kNN or SVM) and the massive scalability of Deep Learning. By condensing infinite datasets into an intelligent, differentiable grid of topological prototypes, FastKMeans guarantees extreme inference speed without sacrificing precision.

---

## 🎯 The Core Philosophy: "Hard Training, Soft Inference"

1. **Phase 1: Hard Stream Learning (EMA)**
   Data flows in infinitely via `fit_batch(gradient_step=False)`. Samples are strictly assigned to the nearest centroid (creating perfect Voronoi cells). Centroids shift geometrically using Exponential Moving Averages (EMA).
2. **Phase 2: Differentiable Fine-tuning**
   The topological grid is unfrozen. Using `fit_batch(gradient_step=True)`, the `Adam` optimizer directly minimizes the Loss function (MSE/CrossEntropy/ASL) by physically moving the centroids via gradient descent.
3. **Phase 3: Soft Inference (Smart KNN)**
   During `.predict()`, the target values of the `top_k` closest prototypes are smoothly blended using Inverse Distance Weighting (`soft_type='scaled' / 'softmax_scaled'`).

---

## 🌊 Stream Learning: The Master Function

At the heart of the entire API lies the `fit_batch` master function. You are no longer constrained by RAM. You can stream billions of rows directly from a database:

```python
model = FastKMeansClassifier()

# 1. Build the Grid
for batch_X, batch_y in infinite_stream:
    model.fit_batch(batch_X, batch_y, gradient_step=False)

# 2. Refine with Gradients
for batch_X, batch_y in infinite_stream:
    model.fit_batch(batch_X, batch_y, gradient_step=True)
```
*(The helper functions `fit()` and `finetune()` are just convenient wrappers around `fit_batch` for static datasets).*

---

## 🛠️ Universal Capabilities & Parameters

### 1. Differentiable Routing (`soft_type` & `target_assignment`)
Controls how centroids react to the input $X$ during both inference and target generation:
*   `hard`: Absolute nearest neighbor mapping (Non-differentiable).
*   `mean`: Global average of the Voronoi cell (Non-differentiable).
*   `scaled`: **Differentiable.** Uses `ReLU(similarity)`.
*   `softmax_scaled`: **Differentiable.** Exponentiated scaling divided by the scalar `temperature`.

⚠️ **Crucial Rule:** If you try to run `.finetune()` or `gradient_step=True` while `soft_type` is set to `hard` or `mean`, the framework will throw a protective `RuntimeError`. Gradients cannot flow through boolean masks!

### 2. Auto Feature Weights (`auto_feature_weights=True`)
When enabled, the framework runs a real-time **ANOVA Variance Analysis**. Features with tight within-cluster variances are automatically boosted, while noisy features are suppressed. During `finetune()`, these weights become `nn.Parameter` and are further optimized by Adam.

### 3. Automatic Learning Rate (CatBoost Style)
If you omit the `lr` argument in `.finetune()`, FastKMeans launches a rapid exponential-LR stress test. It mathematically analyzes the loss curve to pinpoint the optimal learning rate and securely rewinds the weights back to safety before the actual training begins.

### 4. Selective Freezing (`freeze_centroids`)
Need to protect certain learned concepts from shifting?
```python
model.freeze_centroids(mask=my_bool_tensor)
model.add_active_centroid(new_X, new_y) # Dynamically injects an unfrozen prototype
```

---

## 📦 The Models

### 🟢 `FastKMeansClassifier`
Best for Multi-class setups.
*   **LVQ Repulsion:** Uses `repulsion_factor` to forcefully push a centroid away from alien classes, sharpening decision boundaries dynamically.
*   **Negative Sampling:** Accelerates gradients on massive datasets (e.g., 100k classes) by only calculating the loss against a random subset of negative classes, utilizing Log-corrected Sampled Softmax.

### 🟡 `FastMultiLabelKMeansClassifier`
Built for overlapping, multi-tag spaces (NLP, Text, Scene recognition).
*   **Asymmetric Loss (ASL):** Dynamically scales down the gradients of dominant 'Negative' (zero) tags, rescuing the network from gradient starvation.
*   **Gap Prediction Strategy:** By passing `strategy='gap'` in `.predict()`, the model finds the largest sequential drop in tag probabilities to auto-determine *exactly* how many tags a document should receive. Zero thresholds needed.

### 🔵 `FastKMeansRegressor`
The ultimate scalable substitution for standard `kNN`.
*   **2-Stage Stratification:** First, it clusters the $Y$-targets into `k_targets` buckets. Second, it spawns `k_features` within each bucket. 
*   **The Mean Representative:** The very first prototype inside any bucket is explicitly assigned as the pure mathematical mean of that bucket, guaranteeing extreme robustness against outliers.

---
### Performance Asymptotics ⚡
Standard kNN inference runs in $\mathcal{O}(N \times D)$. FastKMeans runs in $\mathcal{O}(K \times D)$, where $K$ is the fixed grid size. When `use_faiss=True` is enabled, complexity plummets to **$\mathcal{O}(\log K)$**, making billion-row predictions practically instantaneous.