"""FastKMeans Suite: Scalable Prototype-based Machine Learning.

This module provides high-performance, GPU-accelerated scalable clustering,
classification, and regression estimators based on moving average topologies,
differentiable prototypes, stream learning capabilities, dynamic feature selection,
sample weight support, and architectural feature masking.
"""

import math
import copy
import logging
from typing import Optional, Union, Any, Dict, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def _check_param(name: str, value: Any, allowed: list) -> None:
    """Strict parameter validation without silent fallbacks.

    Args:
        name (str): The name of the parameter being validated.
        value (Any): The provided value for the parameter.
        allowed (list): A strict list of valid options.

    Raises:
        ValueError: With a detailed summary and list of valid options.
    """
    if value not in allowed:
        raise ValueError(
            f"Invalid configuration for parameter '{name}'.\n"
            f"Received: '{value}' (type: {type(value).__name__})\n"
            f"Valid options are strictly: {allowed}."
        )


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (ASL) for extreme Multi-Label Classification.

    Dynamically down-weights easy negative examples to prevent them from
    overwhelming the gradient in highly sparse multi-label datasets.
    """
    
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05, eps: float = 1e-8):
        """Initializes the Asymmetric Loss metric.
        
        Args:
            gamma_neg (float): Decay factor for negative class predictions.
            gamma_pos (float): Decay factor for positive class predictions.
            clip (float): Probability margin under which negatives are fully discarded.
            eps (float): Small epsilon for numerical stability.
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, probabilities: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the unreduced asymmetric focal loss.

        Args:
            probabilities (torch.Tensor): Predicted probabilities.
            target (torch.Tensor): Ground truth binary matrix.

        Returns:
            torch.Tensor: Computed loss scalar array of shape (batch_size,).
        """
        probabilities = torch.clamp(probabilities, self.eps, 1.0 - self.eps)
        anti_probs = torch.clamp(1.0 - probabilities - self.clip, min=0.0)
        
        pos_weight = (anti_probs ** self.gamma_pos) * target
        neg_weight = (probabilities ** self.gamma_neg) * (1.0 - target)
        
        loss_pos = -pos_weight * torch.log(probabilities)
        loss_neg = -neg_weight * torch.log(anti_probs + self.eps)
        
        return (loss_pos + loss_neg).sum(dim=1)


class BaseFastKMeans(nn.Module):
    """Abstract base class orchestrating the FastKMeans logic.

    Handles low-level GPU orchestration, tensor formatting, dynamic feature importance,
    learning rate discovery, and the dual training mechanics (EMA Topology vs Gradient descent).

    Args:
        distance (str): Metric. Options: 'cosine', 'euclidean', 'l1'.
        dtype (str): Precision type. Options: 'float32', 'float16', 'bfloat16'.
        device (str): Computation device ('cuda' or 'cpu').
        min_weight (float): Minimum node weight for survival.
        truncation_threshold (float): Threshold for sparsity.
        batch_size (int): Size of stream batches.
        top_k (int): K-nearest prototypes for soft inference.
        random_state (int): Seed.
        init_mode (str): Prototype initialization ('random', 'kmeans++').
        temperature (float): Softmax routing temperature scalar.
        soft_type (str): Routing logic ('hard', 'mean', 'scaled', 'softmax_scaled').
        use_faiss (bool): Whether to use FAISS HNSW logic.
        use_compile (bool): Whether to use torch.compile optimizations.
        auto_feature_weights (bool): Enable ANOVA algorithmic feature weighting.
        feature_weights (Optional[torch.Tensor]): Manual feature weights override.
        negative_sampling (Optional[int]): Max negative classes for gradient loss.
        l2_reg (float): Standard L2 regularization penalty applied to centroids.
        diversity_reg (float): Orthogonality / Diversity penalty to prevent mode collapse.
    """

    def __init__(
        self,
        distance: str = 'cosine',
        dtype: str = 'float32',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        min_weight: float = 1e-3,
        truncation_threshold: float = 1e-4,
        batch_size: int = 10240,
        top_k: int = 5,
        random_state: int = 42,
        init_mode: str = 'kmeans++',
        temperature: float = 1.0,
        soft_type: str = 'scaled',
        use_faiss: bool = False,
        use_compile: bool = False,
        auto_feature_weights: bool = False,
        feature_weights: Optional[torch.Tensor] = None,
        negative_sampling: Optional[int] = None,
        l2_reg: float = 0.0,
        diversity_reg: float = 0.0
    ) -> None:
        super().__init__()
        
        distance = distance.lower() if isinstance(distance, str) else distance
        dtype = dtype.lower() if isinstance(dtype, str) else dtype
        init_mode = init_mode.lower() if isinstance(init_mode, str) else init_mode
        soft_type = soft_type.lower() if isinstance(soft_type, str) else soft_type

        _check_param('distance', distance, ['cosine', 'euclidean', 'l1'])
        _check_param('dtype', dtype, ['float32', 'float16', 'bfloat16'])
        _check_param('init_mode', init_mode, ['random', 'kmeans++'])
        _check_param('soft_type', soft_type, ['hard', 'mean', 'scaled', 'softmax_scaled'])

        self.distance = distance
        self.dtype = dtype
        self._device = torch.device(device)
        self.min_weight = min_weight
        self.truncation_threshold = truncation_threshold
        self.batch_size = batch_size
        self.top_k = top_k
        self.random_state = random_state
        self.init_mode = init_mode
        self.temperature = temperature
        self.soft_type = soft_type
        self.use_faiss = use_faiss
        self.use_compile = use_compile
        self.auto_feature_weights = auto_feature_weights
        self.negative_sampling = negative_sampling
        self.l2_reg = l2_reg
        self.diversity_reg = diversity_reg

        dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
        self._torch_dtype = dtype_map[self.dtype]

        self.register_buffer('centroids', torch.empty(0, dtype=self._torch_dtype, device=self._device))
        self.register_buffer('centroid_weights', torch.empty(0, dtype=torch.float32, device=self._device))
        self.register_buffer('centroid_feature_weights', torch.empty(0, dtype=torch.float32, device=self._device))
        
        if feature_weights is not None:
            self.register_buffer('feature_weights', feature_weights.to(self._device, dtype=torch.float32))
            
        self.register_buffer('_fi_var_x', torch.empty(0, device=self._device))
        self.register_buffer('_fi_var_within', torch.empty(0, device=self._device))

        self.learning_rate_ = 1e-4 if distance == 'cosine' else 1e-3
        self._lr_finder_state = {'iter': 0, 'best_loss': float('inf'), 'current_lr': 1e-7, 'losses': [], 'lrs': []}
        
        self._is_initialized = False
        self.frozen_mask: Optional[torch.Tensor] = None
        self._faiss_index = None
        self.optimizer = None
        
        if self.use_compile and hasattr(torch, "compile"):
            self._cdist_compiled = torch.compile(self._compute_distances, mode="reduce-overhead")
        else:
            self._cdist_compiled = self._compute_distances

    def _calculate_optimal_k(self, tensor_sub: torch.Tensor, is_target: bool = False) -> int:
        """Analytically determines the optimal number of prototypes for a given sub-manifold 
        using Geometric Information Dispersion.
        """
        N = tensor_sub.shape[0]
        if N <= 1: 
            return 1

        with torch.no_grad():
            mu = tensor_sub.mean(dim=0, keepdim=True)
            dist_metric = getattr(self, 'target_distance', 'euclidean') if is_target else self.distance
            
            if dist_metric == 'cosine':
                mu_norm = F.normalize(mu, p=2, dim=1)
                t_norm = F.normalize(tensor_sub, p=2, dim=1)
                sim = torch.mm(t_norm, mu_norm.t())
            elif dist_metric == 'euclidean':
                dist2 = torch.sum((tensor_sub - mu)**2, dim=1, keepdim=True)
                sim = 1.0 / (1.0 + dist2)
            else: # l1
                dist = torch.abs(tensor_sub - mu).sum(dim=1, keepdim=True)
                sim = 1.0 / (1.0 + dist)
                
            mean_sim = torch.clamp(sim, 0.0, 1.0).mean().item()
            dispersion = 1.0 - mean_sim
            
            # Analytical bound: MDL upper limit (sqrt(N)) scaled by structural dispersion
            k_opt = math.ceil(math.sqrt(N) * dispersion * 3.0)
            return max(1, min(N, k_opt))

    def _enable_gradients(self) -> list:
        """Injects gradients into structural buffers and returns parameters for the optimizer."""
        self.centroids.requires_grad_(True)
        params = [self.centroids]
        
        if hasattr(self, 'centroid_targets'):
            self.centroid_targets.requires_grad_(True)
            params.append(self.centroid_targets)
            
        if hasattr(self, 'feature_weights'):
            self.feature_weights.requires_grad_(True)
            params.append(self.feature_weights)
            
        return params

    def _disable_gradients(self) -> None:
        """Safely removes gradient requirements from structural buffers."""
        self.centroids.requires_grad_(False)
        if hasattr(self, 'centroid_targets'):
            self.centroid_targets.requires_grad_(False)
        if hasattr(self, 'feature_weights'):
            self.feature_weights.requires_grad_(False)

    def _safe_mm(self, mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
        """Safe matrix multiplication handling sparse tensors and precision upcasting."""
        if mat1.is_sparse or mat1.is_sparse_csr:
            return torch.sparse.mm(mat1.to(torch.float32), mat2.to(torch.float32)).to(self._torch_dtype)
        if mat1.dtype in [torch.float16, torch.bfloat16]:
            return torch.mm(mat1.to(torch.float32), mat2.to(torch.float32)).to(mat1.dtype)
        return torch.mm(mat1, mat2)

    def _compute_distances(self, X_batch: torch.Tensor, C: torch.Tensor, feature_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Core mathematical logic for pairwise distances supporting dynamic feature masking.

        Args:
            X_batch (torch.Tensor): Input features.
            C (torch.Tensor): Centroids matrix.
            feature_mask (Optional[torch.Tensor]): Binary mask matrix (N, D).
            
        Returns:
            torch.Tensor: Pairwise similarity / inverse distance.
        """
        # FAST PATH: Standard 1D Math backward compatibility
        if feature_mask is None and getattr(self, 'feature_weights', None) is None:
            if self.distance == 'cosine':
                X_n = F.normalize(X_batch.to(torch.float32), p=2, dim=1).to(self._torch_dtype)
                C_n = F.normalize(C.to(torch.float32), p=2, dim=1).to(self._torch_dtype)
                return self._safe_mm(X_n, C_n.t())
            elif self.distance == 'euclidean':
                sim = self._safe_mm(X_batch, C.t())
                X_f32, C_f32 = X_batch.to(torch.float32), C.to(torch.float32)
                x2 = torch.sum(X_f32 ** 2, dim=1, keepdim=True)
                c2 = torch.sum(C_f32 ** 2, dim=1)
                dist = torch.clamp(x2 + c2 - 2 * sim.to(torch.float32), min=0.0)
                return (1.0 / (1.0 + dist)).to(self._torch_dtype)
            elif self.distance == 'l1':
                dist = torch.cdist(X_batch.to(torch.float32), C.to(torch.float32), p=1.0)
                return (1.0 / (1.0 + dist)).to(self._torch_dtype)

        # MASKED / WEIGHTED PATH
        if getattr(self, 'feature_weights', None) is not None:
            M = self.feature_weights.clamp(min=0.0).unsqueeze(0)
        else:
            M = torch.ones((1, X_batch.shape[1]), dtype=torch.float32, device=self._device)

        if feature_mask is not None:
            M = M * feature_mask.to(torch.float32)

        if self.distance == 'cosine':
            num = self._safe_mm(X_batch * M, C.t())
            den_X = torch.sqrt((X_batch**2 * M).sum(dim=1, keepdim=True)).clamp(min=1e-9)
            den_C = torch.sqrt(self._safe_mm(M, (C**2).t())).clamp(min=1e-9)
            return num / (den_X * den_C)
            
        elif self.distance == 'euclidean':
            term1 = (X_batch**2 * M).sum(dim=1, keepdim=True)
            term2 = self._safe_mm(M, (C**2).t())
            term3 = 2 * self._safe_mm(X_batch * M, C.t())
            dist2 = torch.clamp(term1 + term2 - term3, min=0.0)
            
            # Normalize to avoid penalizing missing values unfairly
            active_feats = M.sum(dim=1, keepdim=True).clamp(min=1e-9)
            dist2 = dist2 * (X_batch.shape[1] / active_feats)
            return (1.0 / (1.0 + dist2)).to(self._torch_dtype)
            
        elif self.distance == 'l1':
            X_f32 = X_batch.to(torch.float32).unsqueeze(1)
            C_f32 = C.to(torch.float32).unsqueeze(0)
            
            dist = (torch.abs(X_f32 - C_f32) * M.unsqueeze(1)).sum(dim=2)
            
            active_feats = M.sum(dim=1, keepdim=True).clamp(min=1e-9)
            dist = dist * (X_batch.shape[1] / active_feats)
            return (1.0 / (1.0 + dist)).to(self._torch_dtype)

    def _cdist(self, X_batch: torch.Tensor, C: torch.Tensor, feature_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Wrapper around distance computation with a robust fallback."""
        try:
            return self._cdist_compiled(X_batch, C, feature_mask)
        except Exception as e:
            logger.warning(f"PyTorch compilation failed: {e}. Falling back to eager mode.")
            self._cdist_compiled = self._compute_distances
            return self._compute_distances(X_batch, C, feature_mask)

    def _cdist_topk(self, X_batch: torch.Tensor, top_k: int, feature_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates similarities and strictly limits to top-k closest components."""
        sim = self._cdist(X_batch, self.centroids, feature_mask)
        K_total = self.centroids.shape[0]
        actual_k = K_total if (top_k == -1 or top_k >= K_total) else top_k
        
        if actual_k == K_total:
            return sim, torch.arange(K_total, device=self._device).unsqueeze(0).expand(X_batch.shape[0], -1)
            
        topk_vals, topk_indices = torch.topk(sim, k=actual_k, dim=1)
        masked_sim = torch.full_like(sim, -float('inf'))
        masked_sim.scatter_(1, topk_indices, topk_vals)
        return masked_sim, topk_indices

    def _get_soft_routing_scores(self, sim: torch.Tensor) -> torch.Tensor:
        """Centralized routing logic handling target aggregations."""
        if self.soft_type == 'hard':
            max_sim = sim.max(dim=1, keepdim=True)[0]
            return (sim == max_sim).to(self._torch_dtype)
        elif self.soft_type == 'mean':
            mask = (sim != -float('inf')).to(self._torch_dtype)
            return mask / (mask.sum(dim=1, keepdim=True) + 1e-9)
        elif self.soft_type == 'scaled':
            weights = F.relu(sim)
            return weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
        elif self.soft_type == 'softmax_scaled':
            return F.softmax((sim / self.temperature).to(torch.float32), dim=1).to(self._torch_dtype)

    def _get_regularization_loss(self) -> torch.Tensor:
        """Calculates L2 weight decay and Orthogonality (Diversity) constraints."""
        reg = torch.tensor(0.0, device=self._device, dtype=torch.float32)
        
        if self.l2_reg > 0:
            reg += self.l2_reg * torch.sum(self.centroids ** 2)
            
        if self.diversity_reg > 0:
            if self.distance == 'cosine':
                C_norm = F.normalize(self.centroids.to(torch.float32), p=2, dim=1)
                sim = torch.mm(C_norm, C_norm.t())
            elif self.distance == 'euclidean':
                C_f32 = self.centroids.to(torch.float32)
                c2 = torch.sum(C_f32 ** 2, dim=1)
                dist2 = torch.clamp(c2.unsqueeze(1) + c2.unsqueeze(0) - 2 * torch.mm(C_f32, C_f32.t()), min=0.0)
                sim = 1.0 / (1.0 + dist2)
            else:
                dist = torch.cdist(self.centroids.to(torch.float32), self.centroids.to(torch.float32), p=1.0)
                sim = 1.0 / (1.0 + dist)
                
            eye = torch.eye(sim.shape[0], device=self._device)
            # Penalize off-diagonal correlations to prevent Mode Collapse
            reg += self.diversity_reg * torch.sum((sim * (1 - eye)) ** 2)
            
        return reg

    def _prepare_feature_weights(self, X: torch.Tensor) -> None:
        """Safely injects feature weights buffer if auto_weights is enabled."""
        if self.auto_feature_weights and not hasattr(self, "feature_weights"):
            self.register_buffer('feature_weights', torch.ones(X.shape[1], dtype=torch.float32, device=self._device))

    def _format_input(self, X: Any) -> torch.Tensor:
        """Converts arbitrary inputs to optimal PyTorch tensors."""
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X.todense() if hasattr(X, "todense") else X, dtype=self._torch_dtype)
        X = X.to(self._device, dtype=self._torch_dtype)

        if self.distance == 'cosine' and not hasattr(self, 'feature_weights'):
            X = F.normalize(X.to(torch.float32), p=2, dim=1).to(self._torch_dtype)
        return X

    def _format_sample_weight(self, sample_weight: Any, N: int) -> torch.Tensor:
        """Formats sample weights, strictly enforcing non-negativity."""
        if sample_weight is None:
            return torch.ones((N, 1), dtype=torch.float32, device=self._device)
        if not isinstance(sample_weight, torch.Tensor):
            sample_weight = torch.tensor(sample_weight, dtype=torch.float32)
        sw = sample_weight.to(self._device, dtype=torch.float32).view(-1, 1)
        return torch.clamp(sw, min=0.0)

    def _format_feature_mask(self, feature_mask: Any, N: int, D: int) -> Optional[torch.Tensor]:
        """Validates and broadcasts feature masking tensor."""
        if feature_mask is None:
            return None
        if not isinstance(feature_mask, torch.Tensor):
            feature_mask = torch.tensor(feature_mask, dtype=torch.float32)
        fm = feature_mask.to(self._device, dtype=torch.float32)
        if fm.ndim == 1:
            fm = fm.unsqueeze(0).expand(N, -1)
        return torch.clamp(fm, 0.0, 1.0)

    def _split_eval(self, X: torch.Tensor, y: Optional[torch.Tensor], sample_weight: torch.Tensor, feature_mask: Optional[torch.Tensor], eval_set: Optional[Tuple], eval_fraction: float):
        """Generates validation sets dynamically and slices associated mappings."""
        if eval_set is not None:
            X_val = self._format_input(eval_set[0])
            y_val = self._validate_targets(eval_set[1]) if eval_set[1] is not None else None
            sw_val = self._format_sample_weight(eval_set[2] if len(eval_set) > 2 else None, X_val.shape[0])
            fm_val = self._format_feature_mask(eval_set[3] if len(eval_set) > 3 else None, X_val.shape[0], X_val.shape[1])
            return X, y, sample_weight, feature_mask, X_val, y_val, sw_val, fm_val
            
        if eval_fraction > 0.0:
            N = X.shape[0]
            split_idx = int(N * (1 - eval_fraction))
            perm = torch.randperm(N, device=self._device)
            idx_train, idx_val = perm[:split_idx], perm[split_idx:]
            
            y_tr = y[idx_train] if y is not None else None
            y_v = y[idx_val] if y is not None else None
            fm_tr = feature_mask[idx_train] if feature_mask is not None else None
            fm_v = feature_mask[idx_val] if feature_mask is not None else None
            
            return X[idx_train], y_tr, sample_weight[idx_train], fm_tr, X[idx_val], y_v, sample_weight[idx_val], fm_v
            
        return X, y, sample_weight, feature_mask, None, None, None, None

    def freeze_centroids(self, mask: Optional[torch.Tensor] = None) -> None:
        """Freezes selected (or all) centroids, preventing topological and gradient shifts."""
        self.frozen_mask = torch.ones(self.centroids.shape[0], dtype=torch.bool, device=self._device) if mask is None else mask.to(self._device, dtype=torch.bool)

    def accumulate_feature_importance_batch(self, X_batch: torch.Tensor, sample_weight: torch.Tensor, feature_mask: Optional[torch.Tensor] = None) -> None:
        """Processes a single stream chunk for dynamic ANOVA variance feature weighting."""
        with torch.no_grad():
            if not self._is_initialized or self.centroids.shape[0] == 0: 
                return
                
            X_f32 = X_batch.to(torch.float32)
            M = feature_mask if feature_mask is not None else torch.ones_like(X_f32)
            
            sw = sample_weight.flatten()
            M_sw = M * sw.unsqueeze(1)
            
            sum_w = M_sw.sum(dim=0).clamp(min=1e-9)
            mean_x = (X_f32 * M_sw).sum(dim=0) / sum_w
            var_x = (M_sw * (X_f32 - mean_x) ** 2).sum(dim=0) / sum_w
            
            pull_idx = torch.argmax(self._cdist(X_batch, self.centroids, feature_mask), dim=1)
            C_assigned = self.centroids[pull_idx].to(torch.float32)
            
            var_w = (M_sw * (X_f32 - C_assigned) ** 2).sum(dim=0) / sum_w

            if len(self._fi_var_x) == 0:
                self._fi_var_x, self._fi_var_within = var_x, var_w
            else:
                self._fi_var_x = 0.9 * self._fi_var_x + 0.1 * var_x
                self._fi_var_within = 0.9 * self._fi_var_within + 0.1 * var_w

            weights = 1.0 - (self._fi_var_within / (self._fi_var_x + 1e-9))
            if hasattr(self, "feature_weights"):
                self.feature_weights.data = torch.clamp(weights, min=0.0, max=1.0)

    def calculate_feature_importance(self, X: Any, sample_weight: Any = None, feature_mask: Any = None) -> None:
        """Calculates global algorithmic feature weights over the provided dataset."""
        X_t = self._format_input(X)
        N, D = X_t.shape
        sw_t = self._format_sample_weight(sample_weight, N)
        fm_t = self._format_feature_mask(feature_mask, N, D)
        
        self._fi_var_x = torch.empty(0, device=self._device)
        for i in range(0, N, self.batch_size):
            fm_batch = fm_t[i:i+self.batch_size] if fm_t is not None else None
            self.accumulate_feature_importance_batch(X_t[i:i+self.batch_size], sw_t[i:i+self.batch_size], fm_batch)

    def accumulate_learning_rate_batch(self, X_batch: torch.Tensor, y_batch: Optional[torch.Tensor], sample_weight: torch.Tensor, feature_mask: Optional[torch.Tensor], min_lr: float = 1e-7, max_lr: float = 10.0, steps: int = 100) -> None:
        """Single-batch exponential growth step for automatic CatBoost-like LR Finder."""
        if getattr(self, "soft_type", None) in ['hard', 'mean']:
            return
            
        if self.optimizer is None:
            params = self._enable_gradients()
            self.optimizer = optim.Adam(params, lr=min_lr)
            self._lr_finder_state = {'iter': 0, 'best_loss': float('inf'), 'current_lr': min_lr, 'losses': [], 'lrs': []}

        st = self._lr_finder_state
        if st['iter'] >= steps: 
            return

        for param_group in self.optimizer.param_groups: 
            param_group['lr'] = st['current_lr']
            
        loss_val = self._grad_step(X_batch, y_batch, sample_weight, feature_mask, self._get_default_loss_fn(), self.optimizer)
        
        st['losses'].append(loss_val)
        st['lrs'].append(st['current_lr'])
        if loss_val < st['best_loss']: 
            st['best_loss'] = loss_val
            
        if loss_val > st['best_loss'] * 4 or math.isnan(loss_val):
            st['iter'] = steps
        else:
            st['current_lr'] *= (max_lr / min_lr) ** (1 / steps)
            st['iter'] += 1

        if st['iter'] >= steps:
            losses = torch.tensor(st['losses'])
            smoothed = F.avg_pool1d(losses.unsqueeze(0).unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze()
            best_idx = torch.argmax(smoothed[:-1] - smoothed[1:]).item()
            self.learning_rate_ = st['lrs'][best_idx]
            self.optimizer = None
            self._disable_gradients()

    def find_learning_rate(self, X: Any, y: Any = None, sample_weight: Any = None, feature_mask: Any = None) -> None:
        """Finds optimal Learning Rate automatically without corrupting grid topology."""
        if getattr(self, "soft_type", None) in ['hard', 'mean']: 
            return
        
        logger.info("Automatic Learning Rate is calculating...")
        initial_state = copy.deepcopy(self.state_dict())
        
        X_t = self._format_input(X)
        N, D = X_t.shape
        y_t = self._validate_targets(y) if y is not None else None
        sw_t = self._format_sample_weight(sample_weight, N)
        fm_t = self._format_feature_mask(feature_mask, N, D)
        
        for i in range(0, N, self.batch_size):
            y_batch = y_t[i:i+self.batch_size] if y_t is not None else None
            fm_batch = fm_t[i:i+self.batch_size] if fm_t is not None else None
            
            self.accumulate_learning_rate_batch(X_t[i:i+self.batch_size], y_batch, sw_t[i:i+self.batch_size], fm_batch)
            if self._lr_finder_state['iter'] >= 100: 
                break
                
        self.load_state_dict(initial_state)
        self.optimizer = None
        self._disable_gradients()
        logger.info(f"Automatic Learning Rate configured to: {self.learning_rate_:.6f}")

    def fit_batch(self, X: Any, y: Any = None, sample_weight: Any = None, feature_mask: Any = None, gradient_step: bool = False) -> float:
        """Master API capable of Stream processing with Masks and Regularization."""
        X_t = self._format_input(X)
        N, D = X_t.shape
        y_t = self._validate_targets(y) if y is not None else None
        sw_t = self._format_sample_weight(sample_weight, N)
        fm_t = self._format_feature_mask(feature_mask, N, D)
        
        if not self._is_initialized: 
            self._prepare_feature_weights(X_t)
            self._initialize(X_t, y_t)
            
        if self.auto_feature_weights and not gradient_step: 
            self.accumulate_feature_importance_batch(X_t, sw_t, fm_t)
            
        if gradient_step:
            if getattr(self, "soft_type", None) in ['hard', 'mean']:
                raise RuntimeError(
                    f"Gradient optimization is IMPOSSIBLE with soft_type='{self.soft_type}'.\n"
                    "Reason: 'hard' and 'mean' mappings are non-differentiable.\n"
                    "Solution: Switch to 'scaled' or 'softmax_scaled'."
                )

            if getattr(self, "optimizer", None) is None: 
                params = self._enable_gradients()
                self.optimizer = optim.Adam(params, lr=self.learning_rate_)
                
            self.train()
            return self._grad_step(X_t, y_t, sw_t, fm_t, self._get_default_loss_fn(), self.optimizer)
        else:
            self.eval()
            return self._ema_step(X_t, y_t, sw_t, fm_t)

    def fit(self, X: Any, y: Any = None, sample_weight: Any = None, feature_mask: Any = None, 
            max_iters: int = 50, tol: float = 1e-4, eval_set: Optional[Tuple] = None, 
            eval_fraction: float = 0.0, verbose: bool = True) -> 'BaseFastKMeans':
        """Full dataset mapping using hard K-Means exponential moving average mechanics."""
        torch.manual_seed(self.random_state)
        
        X_t = self._format_input(X)
        N, D = X_t.shape
        y_t = self._validate_targets(y) if y is not None else None
        sw_t = self._format_sample_weight(sample_weight, N)
        fm_t = self._format_feature_mask(feature_mask, N, D)
        
        X_train, y_train, sw_train, fm_train, X_val, y_val, sw_val, fm_val = self._split_eval(X_t, y_t, sw_t, fm_t, eval_set, eval_fraction)
        N_train = X_train.shape[0]
        
        pbar = tqdm(range(max_iters), desc="Fit (EMA Topology)") if verbose else range(max_iters)
        for _ in pbar:
            max_shift = 0.0
            perm = torch.randperm(N_train, device=self._device)
            X_shuff = X_train[perm]
            y_shuff = y_train[perm] if y_train is not None else None
            sw_shuff = sw_train[perm]
            fm_shuff = fm_train[perm] if fm_train is not None else None
            
            for i in range(0, N_train, self.batch_size):
                batch_shift = self.fit_batch(
                    X_shuff[i:i+self.batch_size], 
                    y_shuff[i:i+self.batch_size] if y_shuff is not None else None, 
                    sw_shuff[i:i+self.batch_size], 
                    fm_shuff[i:i+self.batch_size] if fm_shuff is not None else None,
                    gradient_step=False
                )
                max_shift = max(max_shift, batch_shift)
                
            if verbose: 
                pbar.set_postfix({"Shift": f"{max_shift:.5f}", "K": len(self.centroids)})
                
            if max_shift < tol: 
                break
        return self

    def finetune(self, X: Any, y: Any = None, sample_weight: Any = None, feature_mask: Any = None, 
                 epochs: int = 10, eval_set: Optional[Tuple] = None, eval_fraction: float = 0.0, 
                 early_stopping_rounds: int = 3, lr: Optional[float] = None, verbose: bool = True) -> 'BaseFastKMeans':
        """Fine-tunes the prototype grid via differentiable optimization."""
        if getattr(self, "soft_type", None) in ['hard', 'mean']:
            raise RuntimeError(f"Cannot execute finetune() with soft_type='{self.soft_type}'.")

        X_t = self._format_input(X)
        N, D = X_t.shape
        y_t = self._validate_targets(y) if y is not None else None
        sw_t = self._format_sample_weight(sample_weight, N)
        fm_t = self._format_feature_mask(feature_mask, N, D)
        
        X_train, y_train, sw_train, fm_train, X_val, y_val, sw_val, fm_val = self._split_eval(X_t, y_t, sw_t, fm_t, eval_set, eval_fraction)
        
        if not self._is_initialized:
            self._prepare_feature_weights(X_train)
            self._initialize(X_train, y_train)
            if self.auto_feature_weights:
                logger.info("Generating algorithmic feature weights via ANOVA before gradient descent...")
                self.calculate_feature_importance(X_train, sw_train, fm_train)
        
        if lr is not None:
            self.learning_rate_ = lr
        elif getattr(self, "optimizer", None) is None:
            self.find_learning_rate(X_train, y_train, sw_train, fm_train)

        params = self._enable_gradients()
        self.optimizer = optim.Adam(params, lr=self.learning_rate_)
        scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        best_score, best_state, no_imp, N_train = -float('inf'), None, 0, X_train.shape[0]

        pbar = tqdm(range(epochs), desc="Finetune (Gradient)") if verbose else range(epochs)
        for epoch in pbar:
            epoch_loss = 0.0
            perm = torch.randperm(N_train, device=self._device)
            X_shuff = X_train[perm]
            y_shuff = y_train[perm] if y_train is not None else None
            sw_shuff = sw_train[perm]
            fm_shuff = fm_train[perm] if fm_train is not None else None
            
            for i in range(0, N_train, self.batch_size):
                loss_step = self.fit_batch(
                    X_shuff[i:i+self.batch_size], 
                    y_shuff[i:i+self.batch_size] if y_shuff is not None else None, 
                    sw_shuff[i:i+self.batch_size], 
                    fm_shuff[i:i+self.batch_size] if fm_shuff is not None else None,
                    gradient_step=True
                )
                epoch_loss += loss_step * min(self.batch_size, N_train - i)

            scheduler.step()
            metrics = {"Loss": f"{epoch_loss / N_train:.4f}", "LR": f"{scheduler.get_last_lr()[0]:.6f}"}
            
            if X_val is not None:
                self.eval()
                val_score, maximize = self._evaluate_metric(X_val, y_val, sw_val, fm_val)
                metrics["Val Score"] = f"{val_score:.4f}"
                
                is_better = (val_score > best_score) if maximize else (val_score < best_score)
                if is_better or best_state is None:
                    best_score, best_state, no_imp = val_score, copy.deepcopy(self.state_dict()), 0
                else:
                    no_imp += 1
                    
                if no_imp >= early_stopping_rounds:
                    if verbose: logger.info(f"Early stopping at epoch {epoch}")
                    self.load_state_dict(best_state)
                    break
                    
            if verbose: pbar.set_postfix(metrics)
            
        self._disable_gradients()
        return self

    # Abstract declarations
    def _validate_targets(self, y: Any) -> Optional[torch.Tensor]: raise NotImplementedError
    def _initialize(self, X: torch.Tensor, y: Optional[torch.Tensor]) -> None: raise NotImplementedError
    def _ema_step(self, X: torch.Tensor, y: Optional[torch.Tensor], sample_weight: torch.Tensor, feature_mask: Optional[torch.Tensor]) -> float: raise NotImplementedError
    def _grad_step(self, X: torch.Tensor, y: Optional[torch.Tensor], sample_weight: torch.Tensor, feature_mask: Optional[torch.Tensor], loss_fn: Callable, optimizer: optim.Optimizer) -> float: raise NotImplementedError
    def forward(self, X: torch.Tensor, feature_mask: Optional[torch.Tensor] = None) -> torch.Tensor: raise NotImplementedError
    def _get_default_loss_fn(self) -> Callable: raise NotImplementedError
    def _evaluate_metric(self, X_val: torch.Tensor, y_val: Optional[torch.Tensor], sample_weight: torch.Tensor, feature_mask: Optional[torch.Tensor]) -> Tuple[float, bool]: raise NotImplementedError


class FastKMeansClusterer(BaseFastKMeans):
    """Pure Unsupervised Clustering with Feature Masking capabilities."""

    def __init__(self, k_init: Union[int, str] = 'auto', **kwargs):
        super().__init__(**kwargs)
        if isinstance(k_init, str): _check_param('k_init', k_init.lower(), ['auto'])
        self.k_init = k_init if isinstance(k_init, int) else k_init.lower()
        
    def _validate_targets(self, y: Any) -> Optional[torch.Tensor]:
        return None

    def _initialize(self, X: torch.Tensor, y: Optional[torch.Tensor]) -> None:
        n_samples = X.shape[0]
        k = self._calculate_optimal_k(X, is_target=False) if self.k_init == 'auto' else min(self.k_init, n_samples)
        
        if self.init_mode == 'random':
            centers = X[torch.randperm(n_samples, device=self._device)[:k]]
        else:
            centers = X[torch.randint(0, n_samples, (1,), device=self._device)]
            for _ in range(1, k):
                sim = self._cdist(X, centers)
                if self.distance == 'cosine':
                    d_x = 1.0 - sim.max(dim=1)[0]
                else:
                    d_x = 1.0 / (sim.max(dim=1)[0] + 1e-9) - 1.0
                
                probs = (d_x.clamp(min=0.0) ** 2).to(torch.float32)
                idx = torch.multinomial(probs, 1) if probs.sum() > 0 else torch.randint(0, n_samples, (1,), device=self._device)
                centers = torch.cat([centers, X[idx]], dim=0)

        self.centroids = centers
        self.centroid_weights = torch.ones(self.centroids.shape[0], dtype=torch.float32, device=self._device)
        self._is_initialized = True

    def forward(self, X: torch.Tensor, feature_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        sim, _ = self._cdist_topk(X, self.top_k, feature_mask)
        return self._get_soft_routing_scores(sim)

    def _get_default_loss_fn(self) -> Callable:
        return lambda x, y: x 

    def _ema_step(self, X_batch: torch.Tensor, y_batch: Optional[torch.Tensor], sample_weight: torch.Tensor, feature_mask: Optional[torch.Tensor]) -> float:
        with torch.no_grad():
            sim = self._cdist(X_batch, self.centroids, feature_mask)
            pull_idx = torch.argmax(sim, dim=1)
            K = self.centroids.shape[0]
            
            sw = sample_weight.flatten()
            W_update = torch.bincount(pull_idx, weights=sw, minlength=K).to(torch.float32)
            valid = (W_update > 0) & (~self.frozen_mask if self.frozen_mask is not None else True)
            
            self.centroid_weights[valid] += W_update[valid]
            
            old_centroids = self.centroids.clone()
            X_batch_f32 = X_batch.to(torch.float32)
            
            if feature_mask is None:
                ema_lr_full = torch.zeros(K, dtype=torch.float32, device=self._device)
                ema_lr_full[valid] = W_update[valid] / self.centroid_weights[valid]
                
                X_pull_sum = torch.zeros((K, X_batch.shape[1]), dtype=torch.float32, device=self._device).index_add_(0, pull_idx, X_batch_f32 * sw.unsqueeze(1))
                ema_lr_exp = ema_lr_full[valid].unsqueeze(1)
                self.centroids[valid] = ((1 - ema_lr_exp) * self.centroids[valid].to(torch.float32) + ema_lr_exp * (X_pull_sum[valid] / W_update[valid].unsqueeze(1))).to(self._torch_dtype)
            else:
                if getattr(self, "centroid_feature_weights", None) is None or self.centroid_feature_weights.numel() == 0:
                    self.centroid_feature_weights = self.centroid_weights.unsqueeze(1).expand(-1, X_batch.shape[1]).clone()
                    
                M_sw = feature_mask * sw.unsqueeze(1)
                W_feat_update = torch.zeros((K, X_batch.shape[1]), dtype=torch.float32, device=self._device).index_add_(0, pull_idx, M_sw)
                self.centroid_feature_weights[valid] += W_feat_update[valid]
                
                ema_lr_feat_full = torch.zeros((K, X_batch.shape[1]), dtype=torch.float32, device=self._device)
                ema_lr_feat_full[valid] = W_feat_update[valid] / self.centroid_feature_weights[valid].clamp(min=1e-9)
                
                X_feat_sum = torch.zeros((K, X_batch.shape[1]), dtype=torch.float32, device=self._device).index_add_(0, pull_idx, X_batch_f32 * M_sw)
                self.centroids[valid] = ((1 - ema_lr_feat_full[valid]) * self.centroids[valid].to(torch.float32) + ema_lr_feat_full[valid] * (X_feat_sum[valid] / W_feat_update[valid].clamp(min=1e-9))).to(self._torch_dtype)
            
            if self.distance == 'cosine': 
                self.centroids = F.normalize(self.centroids.to(torch.float32), p=2, dim=1).to(self._torch_dtype)
                
            return torch.norm(self.centroids.to(torch.float32) - old_centroids.to(torch.float32), dim=1).max().item()

    def _grad_step(self, X_batch: torch.Tensor, y_batch: Optional[torch.Tensor], sample_weight: torch.Tensor, feature_mask: Optional[torch.Tensor], loss_fn: Callable, optimizer: optim.Optimizer) -> float:
        optimizer.zero_grad()
        sim = self._cdist(X_batch, self.centroids, feature_mask)
        
        if self.distance == 'cosine':
            loss = 1.0 - sim.max(dim=1)[0]
        else:
            loss = 1.0 / (sim.max(dim=1)[0] + 1e-9) - 1.0
            
        loss = (loss * sample_weight.flatten()).mean()
        loss = loss + self._get_regularization_loss()
        loss.backward()
        
        if self.frozen_mask is not None: 
            self.centroids.grad[self.frozen_mask] = 0.0
            
        optimizer.step()
        
        if getattr(self, "feature_weights", None) is not None:
            with torch.no_grad(): self.feature_weights.clamp_(0.0, 1.0)
            
        if self.distance == 'cosine':
            with torch.no_grad(): self.centroids.data = F.normalize(self.centroids.data.to(torch.float32), p=2, dim=1).to(self._torch_dtype)
            
        return loss.item()

    def _evaluate_metric(self, X_val: torch.Tensor, y_val: Optional[torch.Tensor], sample_weight: torch.Tensor, feature_mask: Optional[torch.Tensor]) -> Tuple[float, bool]:
        with torch.no_grad():
            sim = self._cdist(X_val, self.centroids, feature_mask)
            if self.distance == 'cosine':
                val = sim.max(dim=1)[0]
            else:
                val = -(1.0 / (sim.max(dim=1)[0] + 1e-9) - 1.0) 
            return (val * sample_weight.flatten()).mean().item(), True

    def predict(self, X: Any, feature_mask: Any = None) -> torch.Tensor:
        with torch.no_grad(): 
            X_t = self._format_input(X)
            fm = self._format_feature_mask(feature_mask, X_t.shape[0], X_t.shape[1])
            return torch.argmax(self._cdist(X_t, self.centroids, fm), dim=1).to(torch.int32)


class FastKMeansClassifier(BaseFastKMeans):
    """Classifier utilizing Negative Sampling, LVQ Repulsion, and Mask-aware Routing."""
    
    def __init__(self, k_init: Union[int, str] = 'auto', repulsion_factor: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        if isinstance(k_init, str): _check_param('k_init', k_init.lower(), ['auto'])
        self.k_init = k_init if isinstance(k_init, int) else k_init.lower()
        self.repulsion_factor = repulsion_factor
        self.register_buffer('centroid_labels', torch.empty(0, dtype=torch.long, device=self._device))
        self.classes_ = torch.empty(0, dtype=torch.long, device=self._device)

    def _validate_targets(self, y: Any) -> torch.Tensor:
        return (y if isinstance(y, torch.Tensor) else torch.tensor(y)).flatten().to(torch.long)

    def _initialize(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.classes_ = torch.unique(y)
        all_c, all_l = [],[]
        for c in self.classes_:
            X_c = X[y == c]
            n_samples = X_c.shape[0]
            k = self._calculate_optimal_k(X_c, is_target=False) if self.k_init == 'auto' else min(self.k_init, n_samples)
            
            if self.init_mode == 'random':
                centers = X_c[torch.randperm(n_samples, device=self._device)[:k]]
            else: 
                centers = X_c[torch.randint(0, n_samples, (1,), device=self._device)]
                for _ in range(1, k):
                    sim = self._cdist(X_c, centers)
                    d_x = (1.0 - sim.max(dim=1)[0]) if self.distance == 'cosine' else (1.0 / (sim.max(dim=1)[0] + 1e-9) - 1.0)
                    probs = (d_x.clamp(min=0.0) ** 2).to(torch.float32)
                    idx = torch.multinomial(probs, 1) if probs.sum() > 0 else torch.randint(0, n_samples, (1,), device=self._device)
                    centers = torch.cat([centers, X_c[idx]], dim=0)
                    
            all_c.append(centers)
            all_l.append(torch.full((k,), c.item(), dtype=torch.long, device=self._device))
            
        self.centroids = torch.cat(all_c, dim=0)
        self.centroid_labels = torch.cat(all_l, dim=0)
        self.centroid_weights = torch.ones(self.centroids.shape[0], dtype=torch.float32, device=self._device)
        self._is_initialized = True

    def forward(self, X: torch.Tensor, feature_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        sim, _ = self._cdist_topk(X, self.top_k, feature_mask)
        scores = self._get_soft_routing_scores(sim)
        
        class_logits = torch.zeros((X.shape[0], len(self.classes_)), dtype=self._torch_dtype, device=self._device)
        label_map = torch.tensor([{c.item(): i for i, c in enumerate(self.classes_)}[lbl.item()] for lbl in self.centroid_labels], device=self._device)
        class_logits.scatter_reduce_(1, label_map.unsqueeze(0).expand(X.shape[0], -1), scores, reduce='sum', include_self=False)
        return class_logits

    def _get_default_loss_fn(self) -> Callable: 
        return nn.CrossEntropyLoss(reduction='none')

    def _ema_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor, sample_weight: torch.Tensor, feature_mask: Optional[torch.Tensor]) -> float:
        with torch.no_grad():
            sim = self._cdist(X_batch, self.centroids, feature_mask)
            mask_same = (y_batch.unsqueeze(1) == self.centroid_labels.unsqueeze(0))
            K = self.centroids.shape[0]
            
            pull_sim = torch.where(mask_same, sim, torch.tensor(-float('inf'), device=self._device))
            pull_idx = torch.argmax(pull_sim, dim=1)
            
            sw = sample_weight.flatten()
            W_update = torch.bincount(pull_idx, weights=sw, minlength=K).to(torch.float32)
            valid = (W_update > 0) & (~self.frozen_mask if self.frozen_mask is not None else True)
            
            self.centroid_weights[valid] += W_update[valid]
            
            old_centroids = self.centroids.clone()
            X_batch_f32 = X_batch.to(torch.float32)

            if feature_mask is None:
                ema_lr_full = torch.zeros(K, dtype=torch.float32, device=self._device)
                ema_lr_full[valid] = W_update[valid] / self.centroid_weights[valid]
                
                X_pull_sum = torch.zeros((K, X_batch.shape[1]), dtype=torch.float32, device=self._device).index_add_(0, pull_idx, X_batch_f32 * sw.unsqueeze(1))
                ema_lr_exp = ema_lr_full[valid].unsqueeze(1)
                self.centroids[valid] = ((1 - ema_lr_exp) * self.centroids[valid].to(torch.float32) + ema_lr_exp * (X_pull_sum[valid] / W_update[valid].unsqueeze(1))).to(self._torch_dtype)
                
                if self.repulsion_factor > 0:
                    push_idx = torch.argmax(torch.where(~mask_same, sim, torch.tensor(-float('inf'), device=self._device)), dim=1)
                    W_push = torch.bincount(push_idx, weights=sw, minlength=K).to(torch.float32)
                    valid_push = (W_push > 0) & valid 
                    if valid_push.any():
                        X_push_sum = torch.zeros((K, X_batch.shape[1]), dtype=torch.float32, device=self._device).index_add_(0, push_idx, X_batch_f32 * sw.unsqueeze(1))
                        ema_lr_push_full = torch.zeros(K, dtype=torch.float32, device=self._device)
                        ema_lr_push_full[valid_push] = W_push[valid_push] / self.centroid_weights[valid_push].clamp(min=1e-9)
                        self.centroids[valid_push] -= (ema_lr_push_full[valid_push].unsqueeze(1) * self.repulsion_factor) * (X_push_sum[valid_push] / W_push[valid_push].unsqueeze(1))
            else:
                if getattr(self, "centroid_feature_weights", None) is None or self.centroid_feature_weights.numel() == 0:
                    self.centroid_feature_weights = self.centroid_weights.unsqueeze(1).expand(-1, X_batch.shape[1]).clone()
                    
                M_sw = feature_mask * sw.unsqueeze(1)
                W_feat_update = torch.zeros((K, X_batch.shape[1]), dtype=torch.float32, device=self._device).index_add_(0, pull_idx, M_sw)
                self.centroid_feature_weights[valid] += W_feat_update[valid]
                
                ema_lr_feat_full = torch.zeros((K, X_batch.shape[1]), dtype=torch.float32, device=self._device)
                ema_lr_feat_full[valid] = W_feat_update[valid] / self.centroid_feature_weights[valid].clamp(min=1e-9)
                
                X_feat_sum = torch.zeros((K, X_batch.shape[1]), dtype=torch.float32, device=self._device).index_add_(0, pull_idx, X_batch_f32 * M_sw)
                self.centroids[valid] = ((1 - ema_lr_feat_full[valid]) * self.centroids[valid].to(torch.float32) + ema_lr_feat_full[valid] * (X_feat_sum[valid] / W_feat_update[valid].clamp(min=1e-9))).to(self._torch_dtype)
                
                if self.repulsion_factor > 0:
                    push_idx = torch.argmax(torch.where(~mask_same, sim, torch.tensor(-float('inf'), device=self._device)), dim=1)
                    W_feat_push = torch.zeros((K, X_batch.shape[1]), dtype=torch.float32, device=self._device).index_add_(0, push_idx, M_sw)
                    valid_push = (W_feat_push > 0).any(dim=1) & valid
                    if valid_push.any():
                        X_feat_push_sum = torch.zeros((K, X_batch.shape[1]), dtype=torch.float32, device=self._device).index_add_(0, push_idx, X_batch_f32 * M_sw)
                        ema_lr_feat_push_full = torch.zeros((K, X_batch.shape[1]), dtype=torch.float32, device=self._device)
                        ema_lr_feat_push_full[valid_push] = W_feat_push[valid_push] / self.centroid_feature_weights[valid_push].clamp(min=1e-9)
                        self.centroids[valid_push] -= (ema_lr_feat_push_full[valid_push] * self.repulsion_factor) * (X_feat_push_sum[valid_push] / W_feat_push[valid_push].clamp(min=1e-9))

            if self.distance == 'cosine': 
                self.centroids = F.normalize(self.centroids.to(torch.float32), p=2, dim=1).to(self._torch_dtype)
                
            return torch.norm(self.centroids.to(torch.float32) - old_centroids.to(torch.float32), dim=1).max().item()

    def _grad_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor, sample_weight: torch.Tensor, feature_mask: Optional[torch.Tensor], loss_fn: Callable, optimizer: optim.Optimizer) -> float:
        optimizer.zero_grad()
        y_mapped = torch.tensor([{c.item(): i for i, c in enumerate(self.classes_)}[lbl.item()] for lbl in y_batch], device=self._device)
        logits = self.forward(X_batch, feature_mask).to(torch.float32)

        if self.negative_sampling is not None and self.negative_sampling < len(self.classes_):
            batch_classes = torch.unique(y_mapped)
            all_classes = torch.arange(len(self.classes_), device=self._device)
            neg_classes = all_classes[~torch.isin(all_classes, batch_classes)]
            
            if len(neg_classes) > self.negative_sampling:
                sampled_neg = neg_classes[torch.randperm(len(neg_classes), device=self._device)[:self.negative_sampling]]
                active_classes = torch.cat([batch_classes, sampled_neg])
                
                correction = math.log(len(neg_classes) / self.negative_sampling)
                mask = torch.zeros_like(logits, dtype=torch.bool)
                mask[:, active_classes] = True
                
                logits[~mask] = -1e9
                logits[:, sampled_neg] += correction

        loss = loss_fn(logits, y_mapped)
        loss = (loss * sample_weight.flatten()).mean()
        loss = loss + self._get_regularization_loss()
        loss.backward()
        
        if self.frozen_mask is not None: 
            self.centroids.grad[self.frozen_mask] = 0.0
            
        optimizer.step()
        
        if getattr(self, "feature_weights", None) is not None:
            with torch.no_grad(): self.feature_weights.clamp_(0.0, 1.0)
            
        if self.distance == 'cosine':
            with torch.no_grad(): self.centroids.data = F.normalize(self.centroids.data.to(torch.float32), p=2, dim=1).to(self._torch_dtype)
                
        return loss.item()

    def _evaluate_metric(self, X_val: torch.Tensor, y_val: torch.Tensor, sample_weight: torch.Tensor, feature_mask: Optional[torch.Tensor]) -> Tuple[float, bool]:
        with torch.no_grad():
            preds = torch.argmax(self.forward(X_val, feature_mask), dim=1)
            y_mapped = torch.tensor([{c.item(): i for i, c in enumerate(self.classes_)}[lbl.item()] for lbl in y_val], device=self._device)
            acc = ((preds == y_mapped).float() * sample_weight.flatten()).sum() / sample_weight.sum()
            return acc.item(), True

    def predict(self, X: Any, feature_mask: Any = None) -> torch.Tensor:
        with torch.no_grad(): 
            X_t = self._format_input(X)
            fm = self._format_feature_mask(feature_mask, X_t.shape[0], X_t.shape[1])
            return self.classes_[torch.argmax(self.forward(X_t, fm), dim=1)].to(torch.int32)


class FastMultiLabelKMeansClassifier(BaseFastKMeans):
    """Multi-Label Classifier featuring Independent Tag Sub-topologies and Configurable ASL."""
    
    def __init__(self, k_init: Union[int, str] = 'auto', repulsion_factor: float = 0.0, 
                 asl_gamma_neg: float = 4.0, asl_gamma_pos: float = 1.0, 
                 asl_clip: float = 0.05, asl_eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        if isinstance(k_init, str): _check_param('k_init', k_init.lower(), ['auto'])
        self.k_init = k_init if isinstance(k_init, int) else k_init.lower()
        self.repulsion_factor = repulsion_factor
        self.asl_gamma_neg = asl_gamma_neg
        self.asl_gamma_pos = asl_gamma_pos
        self.asl_clip = asl_clip
        self.asl_eps = asl_eps
        
        self.register_buffer('centroid_labels', torch.empty(0, dtype=torch.long, device=self._device))

    def _validate_targets(self, Y: Any) -> torch.Tensor:
        return Y if isinstance(Y, torch.Tensor) else torch.tensor(Y, dtype=torch.float32)

    def _initialize(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        self.classes_ = torch.arange(Y.shape[1], device=self._device)
        all_c, all_l = [],[]
        
        for c in self.classes_:
            X_c = X[torch.where(Y[:, c] > 0)[0]]
            if len(X_c) == 0: continue
            
            k = self._calculate_optimal_k(X_c, is_target=False) if self.k_init == 'auto' else min(self.k_init, X_c.shape[0])
            if self.init_mode == 'random':
                centers = X_c[torch.randperm(X_c.shape[0], device=self._device)[:k]]
            else:
                centers = X_c[torch.randint(0, X_c.shape[0], (1,), device=self._device)]
                for _ in range(1, k):
                    sim = self._cdist(X_c, centers)
                    d_x = (1.0 - sim.max(dim=1)[0]) if self.distance == 'cosine' else (1.0 / (sim.max(dim=1)[0] + 1e-9) - 1.0)
                    probs = (d_x.clamp(min=0.0) ** 2).to(torch.float32)
                    idx = torch.multinomial(probs, 1) if probs.sum() > 0 else torch.randint(0, X_c.shape[0], (1,), device=self._device)
                    centers = torch.cat([centers, X_c[idx]], dim=0)
                    
            all_c.append(centers)
            all_l.append(torch.full((k,), c.item(), dtype=torch.long, device=self._device))
            
        self.centroids = torch.cat(all_c, dim=0)
        self.centroid_labels = torch.cat(all_l, dim=0)
        self.centroid_weights = torch.ones(self.centroids.shape[0], dtype=torch.float32, device=self._device)
        self._is_initialized = True

    def forward(self, X: torch.Tensor, feature_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        sim, _ = self._cdist_topk(X, self.top_k, feature_mask)
        if self.distance == 'cosine': 
            sim = (sim + 1.0) / 2.0 
            
        class_probs = torch.zeros((X.shape[0], len(self.classes_)), dtype=self._torch_dtype, device=self._device)
        class_probs.scatter_reduce_(1, self.centroid_labels.unsqueeze(0).expand(X.shape[0], -1), sim, reduce='amax', include_self=False)
        return torch.clamp(class_probs, 0.0, 1.0)

    def _get_default_loss_fn(self) -> Callable: 
        return AsymmetricLoss(gamma_neg=self.asl_gamma_neg, gamma_pos=self.asl_gamma_pos, 
                              clip=self.asl_clip, eps=self.asl_eps)

    def _ema_step(self, X_batch: torch.Tensor, Y_batch: torch.Tensor, sample_weight: torch.Tensor, feature_mask: Optional[torch.Tensor]) -> float:
        with torch.no_grad():
            K = self.centroids.shape[0]
            sim = self._cdist(X_batch, self.centroids, feature_mask)
            X_batch_f32 = X_batch.to(torch.float32)
            sw = sample_weight.flatten()
            
            if feature_mask is None:
                W_update = torch.zeros(K, dtype=torch.float32, device=self._device)
                X_sum = torch.zeros((K, X_batch.shape[1]), dtype=torch.float32, device=self._device)
                W_push = torch.zeros(K, dtype=torch.float32, device=self._device)
                X_push_sum = torch.zeros((K, X_batch.shape[1]), dtype=torch.float32, device=self._device)
                
                for c in self.classes_:
                    c_mask = (self.centroid_labels == c)
                    if not c_mask.any(): continue
                    
                    doc_mask = (Y_batch[:, c] > 0)
                    if doc_mask.any():
                        best_sub_idx = torch.argmax(sim[doc_mask][:, c_mask], dim=1)
                        best_global_idx = torch.where(c_mask)[0][best_sub_idx]
                        sw_pos = sw[doc_mask]
                        W_update.index_add_(0, best_global_idx, sw_pos)
                        X_sum.index_add_(0, best_global_idx, X_batch_f32[doc_mask] * sw_pos.unsqueeze(1))

                    if self.repulsion_factor > 0:
                        neg_doc_mask = (Y_batch[:, c] <= 0)
                        if neg_doc_mask.any():
                            push_sub_idx = torch.argmax(sim[neg_doc_mask][:, c_mask], dim=1)
                            push_global_idx = torch.where(c_mask)[0][push_sub_idx]
                            sw_neg = sw[neg_doc_mask]
                            W_push.index_add_(0, push_global_idx, sw_neg)
                            X_push_sum.index_add_(0, push_global_idx, X_batch_f32[neg_doc_mask] * sw_neg.unsqueeze(1))

                valid = (W_update > 0) & (~self.frozen_mask if self.frozen_mask is not None else True)
                self.centroid_weights[valid] += W_update[valid]
                
                ema_lr_full = torch.zeros(K, dtype=torch.float32, device=self._device)
                ema_lr_full[valid] = W_update[valid] / self.centroid_weights[valid]
                
                old_centroids = self.centroids.clone()
                self.centroids[valid] = ((1 - ema_lr_full[valid].unsqueeze(1)) * self.centroids[valid].to(torch.float32) + ema_lr_full[valid].unsqueeze(1) * (X_sum[valid] / W_update[valid].unsqueeze(1))).to(self._torch_dtype)
                
                if self.repulsion_factor > 0:
                    valid_push = (W_push > 0) & (~self.frozen_mask if self.frozen_mask is not None else True)
                    if valid_push.any():
                        ema_lr_push_full = torch.zeros(K, dtype=torch.float32, device=self._device)
                        ema_lr_push_full[valid_push] = W_push[valid_push] / self.centroid_weights[valid_push].clamp(min=1e-9)
                        self.centroids[valid_push] -= (ema_lr_push_full[valid_push].unsqueeze(1) * self.repulsion_factor) * (X_push_sum[valid_push] / W_push[valid_push].unsqueeze(1))
                        
            else:
                if getattr(self, "centroid_feature_weights", None) is None or self.centroid_feature_weights.numel() == 0:
                    self.centroid_feature_weights = self.centroid_weights.unsqueeze(1).expand(-1, X_batch.shape[1]).clone()
                    
                M_sw = feature_mask * sw.unsqueeze(1)
                W_feat_update = torch.zeros((K, X_batch.shape[1]), dtype=torch.float32, device=self._device)
                X_feat_sum = torch.zeros((K, X_batch.shape[1]), dtype=torch.float32, device=self._device)
                
                W_feat_push = torch.zeros((K, X_batch.shape[1]), dtype=torch.float32, device=self._device)
                X_feat_push = torch.zeros((K, X_batch.shape[1]), dtype=torch.float32, device=self._device)
                
                for c in self.classes_:
                    c_mask = (self.centroid_labels == c)
                    if not c_mask.any(): continue
                    
                    doc_mask = (Y_batch[:, c] > 0)
                    if doc_mask.any():
                        best_sub_idx = torch.argmax(sim[doc_mask][:, c_mask], dim=1)
                        best_global_idx = torch.where(c_mask)[0][best_sub_idx]
                        M_sw_pos = M_sw[doc_mask]
                        W_feat_update.index_add_(0, best_global_idx, M_sw_pos)
                        X_feat_sum.index_add_(0, best_global_idx, X_batch_f32[doc_mask] * M_sw_pos)

                    if self.repulsion_factor > 0:
                        neg_doc_mask = (Y_batch[:, c] <= 0)
                        if neg_doc_mask.any():
                            push_sub_idx = torch.argmax(sim[neg_doc_mask][:, c_mask], dim=1)
                            push_global_idx = torch.where(c_mask)[0][push_sub_idx]
                            M_sw_neg = M_sw[neg_doc_mask]
                            W_feat_push.index_add_(0, push_global_idx, M_sw_neg)
                            X_feat_push.index_add_(0, push_global_idx, X_batch_f32[neg_doc_mask] * M_sw_neg)

                valid = (W_feat_update > 0).any(dim=1) & (~self.frozen_mask if self.frozen_mask is not None else True)
                self.centroid_feature_weights[valid] += W_feat_update[valid]
                
                ema_lr_feat_full = torch.zeros((K, X_batch.shape[1]), dtype=torch.float32, device=self._device)
                ema_lr_feat_full[valid] = W_feat_update[valid] / self.centroid_feature_weights[valid].clamp(min=1e-9)
                
                old_centroids = self.centroids.clone()
                self.centroids[valid] = ((1 - ema_lr_feat_full[valid]) * self.centroids[valid].to(torch.float32) + ema_lr_feat_full[valid] * (X_feat_sum[valid] / W_feat_update[valid].clamp(min=1e-9))).to(self._torch_dtype)
                
                if self.repulsion_factor > 0:
                    valid_push = (W_feat_push > 0).any(dim=1) & (~self.frozen_mask if self.frozen_mask is not None else True)
                    if valid_push.any():
                        ema_lr_feat_push_full = torch.zeros((K, X_batch.shape[1]), dtype=torch.float32, device=self._device)
                        ema_lr_feat_push_full[valid_push] = W_feat_push[valid_push] / self.centroid_feature_weights[valid_push].clamp(min=1e-9)
                        self.centroids[valid_push] -= (ema_lr_feat_push_full[valid_push] * self.repulsion_factor) * (X_feat_push[valid_push] / W_feat_push[valid_push].clamp(min=1e-9))
            
            if self.distance == 'cosine': 
                self.centroids = F.normalize(self.centroids.to(torch.float32), p=2, dim=1).to(self._torch_dtype)
                
            return torch.norm(self.centroids.to(torch.float32) - old_centroids.to(torch.float32), dim=1).max().item()

    def _grad_step(self, X_batch: torch.Tensor, Y_batch: torch.Tensor, sample_weight: torch.Tensor, feature_mask: Optional[torch.Tensor], loss_fn: Callable, optimizer: optim.Optimizer) -> float:
        optimizer.zero_grad()
        loss = loss_fn(self.forward(X_batch, feature_mask).to(torch.float32), Y_batch.to(torch.float32))
        loss = (loss * sample_weight.flatten()).mean()
        loss = loss + self._get_regularization_loss()
        loss.backward()
        
        if self.frozen_mask is not None: 
            self.centroids.grad[self.frozen_mask] = 0.0
            
        optimizer.step()
        
        if getattr(self, "feature_weights", None) is not None:
            with torch.no_grad(): self.feature_weights.clamp_(0.0, 1.0)
            
        if self.distance == 'cosine':
            with torch.no_grad(): self.centroids.data = F.normalize(self.centroids.data.to(torch.float32), p=2, dim=1).to(self._torch_dtype)
            
        return loss.item()

    def _evaluate_metric(self, X_val: torch.Tensor, y_val: torch.Tensor, sample_weight: torch.Tensor, feature_mask: Optional[torch.Tensor]) -> Tuple[float, bool]:
        with torch.no_grad(): 
            preds = (self.forward(X_val, feature_mask) > 0.5).int()
            acc = ((preds == y_val.int()).float().mean(dim=1) * sample_weight.flatten()).sum() / sample_weight.sum()
            return acc.item(), True

    def predict(self, X: Any, feature_mask: Any = None, strategy: str = 'adaptive', top_k: int = 1) -> torch.Tensor:
        """Translates smooth probabilities into crisp multi-label binary vectors."""
        with torch.no_grad():
            X_t = self._format_input(X)
            fm = self._format_feature_mask(feature_mask, X_t.shape[0], X_t.shape[1])
            scores = self.forward(X_t, fm)
            preds = torch.zeros_like(scores)
            
            if strategy == 'gap':
                if len(self.classes_) <= 1:
                    preds = (scores > 0.5).float()
                else:
                    sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=1)
                    diffs = sorted_scores[:, :-1] - sorted_scores[:, 1:]
                    best_cutoffs = torch.argmax(diffs, dim=1) + 1
                    for i in range(scores.shape[0]):
                        keep = sorted_indices[i, :best_cutoffs[i]]
                        preds[i, keep[scores[i, keep] > 1e-4]] = 1.0
            elif strategy == 'top_k':
                preds.scatter_(1, torch.topk(scores, k=top_k, dim=1)[1], 1.0)
            else:
                preds = (scores > 0.5).float()
                
            return preds.to(torch.int32)


class FastKMeansRegressor(BaseFastKMeans):
    """Memory-efficient Regressor mapping arbitrary input features to Vector targets."""
    
    def __init__(self, k_targets: Union[int, str] = 'auto', k_features: Union[int, str] = 'auto', target_distance: str = 'euclidean', target_assignment: str = 'scaled', **kwargs):
        super().__init__(**kwargs)
        if isinstance(k_targets, str): _check_param('k_targets', k_targets.lower(), ['auto'])
        if isinstance(k_features, str): _check_param('k_features', k_features.lower(), ['auto'])
        self.k_targets = k_targets if isinstance(k_targets, int) else k_targets.lower()
        self.k_features = k_features if isinstance(k_features, int) else k_features.lower()
        
        target_distance = target_distance.lower() if isinstance(target_distance, str) else target_distance
        target_assignment = target_assignment.lower() if isinstance(target_assignment, str) else target_assignment
        
        _check_param('target_distance', target_distance, ['euclidean', 'cosine'])
        _check_param('target_assignment', target_assignment, ['hard', 'mean', 'scaled', 'softmax_scaled'])
        
        self.target_distance = target_distance
        self.target_assignment = target_assignment
        
        self.register_buffer('centroid_targets', torch.empty(0, dtype=self._torch_dtype, device=self._device))

    def _validate_targets(self, y: Any) -> torch.Tensor:
        if y is None: raise ValueError("Regression estimators require target values (y).")
        y_t = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)
        if y_t.ndim == 1:
            return y_t.unsqueeze(1)
        return y_t.to(torch.float32)

    def _initialize(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Executes 2-Stage Target-Feature Stratification."""
        n_samples = X.shape[0]
        k_t_actual = self._calculate_optimal_k(y, is_target=True) if self.k_targets == 'auto' else min(self.k_targets, n_samples)
        
        if self.init_mode == 'random':
            y_c = y[torch.randperm(n_samples, device=self._device)[:k_t_actual]]
        else:
            y_c = y[torch.randint(0, n_samples, (1,), device=self._device)]
            for _ in range(1, k_t_actual):
                if self.target_distance == 'euclidean':
                    dists = torch.cdist(y.to(torch.float32), y_c.to(torch.float32), p=2.0).min(dim=1)[0]
                else:
                    sim_y = torch.mm(F.normalize(y.to(torch.float32), p=2, dim=1), F.normalize(y_c.to(torch.float32), p=2, dim=1).t())
                    dists = (1.0 - sim_y).clamp(min=0).min(dim=1)[0]
                    
                probs = dists ** 2
                idx = torch.multinomial(probs, 1) if probs.sum() > 0 else torch.randint(0, n_samples, (1,), device=self._device)
                y_c = torch.cat([y_c, y[idx]], dim=0)

        if self.target_distance == 'euclidean':
            y_labels = torch.argmin(torch.cdist(y.to(torch.float32), y_c.to(torch.float32), p=2.0), dim=1)
        else:
            sim_y = torch.mm(F.normalize(y.to(torch.float32), p=2, dim=1), F.normalize(y_c.to(torch.float32), p=2, dim=1).t())
            y_labels = torch.argmax(sim_y, dim=1)
        
        all_x_c, all_x_t = [],[]
        
        for c_idx in range(k_t_actual):
            mask = (y_labels == c_idx)
            if not mask.any(): continue
            
            X_sub, y_sub = X[mask], y[mask]
            k_f_actual = self._calculate_optimal_k(X_sub, is_target=False) if self.k_features == 'auto' else min(self.k_features, X_sub.shape[0])
            
            centers_x = X_sub.mean(dim=0, keepdim=True)
            centers_y = y_sub.mean(dim=0, keepdim=True)
            
            if k_f_actual > 1:
                if self.init_mode == 'random':
                    idx = torch.randperm(X_sub.shape[0], device=self._device)[:k_f_actual - 1]
                    centers_x = torch.cat([centers_x, X_sub[idx]], dim=0)
                    centers_y = torch.cat([centers_y, y_sub[idx]], dim=0)
                else: 
                    for _ in range(1, k_f_actual):
                        sim = self._cdist(X_sub, centers_x)
                        d_x = (1.0 - sim.max(dim=1)[0]) if self.distance == 'cosine' else (1.0 / (sim.max(dim=1)[0] + 1e-9) - 1.0)
                        probs = (d_x.clamp(min=0.0) ** 2).to(torch.float32)
                        idx = torch.multinomial(probs, 1) if probs.sum() > 0 else torch.randint(0, X_sub.shape[0], (1,), device=self._device)
                        centers_x = torch.cat([centers_x, X_sub[idx]], dim=0)
                        centers_y = torch.cat([centers_y, y_sub[idx]], dim=0)
            
            all_x_c.append(centers_x)
            all_x_t.append(centers_y)

        self.centroids = torch.cat(all_x_c, dim=0)
        self.centroid_targets = torch.cat(all_x_t, dim=0)
        self.centroid_weights = torch.ones(self.centroids.shape[0], dtype=torch.float32, device=self._device)
        self._is_initialized = True

    def forward(self, X: torch.Tensor, feature_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        sim, _ = self._cdist_topk(X, self.top_k, feature_mask)
        probs = self._get_soft_routing_scores(sim)
        return self._safe_mm(probs.to(self._torch_dtype), self.centroid_targets)

    def _get_default_loss_fn(self) -> Callable: 
        return nn.MSELoss(reduction='none')

    def _ema_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor, sample_weight: torch.Tensor, feature_mask: Optional[torch.Tensor]) -> float:
        with torch.no_grad():
            sim = self._cdist(X_batch, self.centroids, feature_mask)
            pull_idx = torch.argmax(sim, dim=1)
            K = self.centroids.shape[0]
            
            sw = sample_weight.flatten()
            W_update = torch.bincount(pull_idx, weights=sw, minlength=K).to(torch.float32)
            valid = (W_update > 0) & (~self.frozen_mask if self.frozen_mask is not None else True)
            
            self.centroid_weights[valid] += W_update[valid]
            
            old_centroids = self.centroids.clone()
            X_batch_f32 = X_batch.to(torch.float32)
            
            if feature_mask is None:
                ema_lr_full = torch.zeros(K, dtype=torch.float32, device=self._device)
                ema_lr_full[valid] = W_update[valid] / self.centroid_weights[valid]
                
                X_sum = torch.zeros((K, X_batch.shape[1]), dtype=torch.float32, device=self._device).index_add_(0, pull_idx, X_batch_f32 * sw.unsqueeze(1))
                ema_lr_exp = ema_lr_full[valid].unsqueeze(1)
                self.centroids[valid] = ((1 - ema_lr_exp) * self.centroids[valid].to(torch.float32) + ema_lr_exp * (X_sum[valid] / W_update[valid].unsqueeze(1))).to(self._torch_dtype)
            else:
                if getattr(self, "centroid_feature_weights", None) is None or self.centroid_feature_weights.numel() == 0:
                    self.centroid_feature_weights = self.centroid_weights.unsqueeze(1).expand(-1, X_batch.shape[1]).clone()
                    
                M_sw = feature_mask * sw.unsqueeze(1)
                W_feat_update = torch.zeros((K, X_batch.shape[1]), dtype=torch.float32, device=self._device).index_add_(0, pull_idx, M_sw)
                self.centroid_feature_weights[valid] += W_feat_update[valid]
                
                ema_lr_feat_full = torch.zeros((K, X_batch.shape[1]), dtype=torch.float32, device=self._device)
                ema_lr_feat_full[valid] = W_feat_update[valid] / self.centroid_feature_weights[valid].clamp(min=1e-9)
                
                X_feat_sum = torch.zeros((K, X_batch.shape[1]), dtype=torch.float32, device=self._device).index_add_(0, pull_idx, X_batch_f32 * M_sw)
                self.centroids[valid] = ((1 - ema_lr_feat_full[valid]) * self.centroids[valid].to(torch.float32) + ema_lr_feat_full[valid] * (X_feat_sum[valid] / W_feat_update[valid].clamp(min=1e-9))).to(self._torch_dtype)

            Y_f32 = y_batch.to(torch.float32)
            
            # Using the pre-calculated global full-sized array for safety
            ema_lr_targ_full = torch.zeros(K, dtype=torch.float32, device=self._device)
            ema_lr_targ_full[valid] = W_update[valid] / self.centroid_weights[valid]
            ema_lr_targ_exp = ema_lr_targ_full[valid].unsqueeze(1)
            
            if self.target_assignment == 'hard':
                batch_target = Y_f32[torch.argmax(sim, dim=0)][valid]
            elif self.target_assignment == 'mean':
                Y_sum = torch.zeros((K, Y_f32.shape[1]), dtype=torch.float32, device=self._device).index_add_(0, pull_idx, Y_f32 * sw.unsqueeze(1))
                batch_target = Y_sum[valid] / W_update[valid].unsqueeze(1)
            else: 
                P_sim = sim[torch.arange(X_batch.shape[0], device=self._device), pull_idx]
                if self.target_assignment == 'scaled': 
                    P_sim = F.relu(P_sim)
                elif self.target_assignment == 'softmax_scaled': 
                    P_sim = torch.exp(P_sim / self.temperature)
                
                weighted_P_sim = P_sim * sw
                Y_sum = torch.zeros((K, Y_f32.shape[1]), dtype=torch.float32, device=self._device).index_add_(0, pull_idx, Y_f32 * weighted_P_sim.unsqueeze(1))
                
                Sim_sum = torch.zeros(K, dtype=torch.float32, device=self._device).index_add_(0, pull_idx, weighted_P_sim)
                batch_target = Y_sum[valid] / (Sim_sum[valid].unsqueeze(1) + 1e-9)

            self.centroid_targets[valid] = ((1 - ema_lr_targ_exp) * self.centroid_targets[valid].to(torch.float32) + ema_lr_targ_exp * batch_target).to(self._torch_dtype)
            
            if self.distance == 'cosine': 
                self.centroids = F.normalize(self.centroids.to(torch.float32), p=2, dim=1).to(self._torch_dtype)
            
            return torch.norm(self.centroids.to(torch.float32) - old_centroids.to(torch.float32), dim=1).max().item()

    def _grad_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor, sample_weight: torch.Tensor, feature_mask: Optional[torch.Tensor], loss_fn: Callable, optimizer: optim.Optimizer) -> float:
        optimizer.zero_grad()
        loss = loss_fn(self.forward(X_batch, feature_mask).to(torch.float32), y_batch.to(torch.float32))
        loss = (loss.mean(dim=1) * sample_weight.flatten()).mean()
        loss = loss + self._get_regularization_loss()
        loss.backward()
        
        if self.frozen_mask is not None:
            self.centroids.grad[self.frozen_mask] = 0.0
            self.centroid_targets.grad[self.frozen_mask] = 0.0
            
        optimizer.step()
        
        if getattr(self, "feature_weights", None) is not None:
            with torch.no_grad(): self.feature_weights.clamp_(0.0, 1.0)
            
        if self.distance == 'cosine':
            with torch.no_grad(): self.centroids.data = F.normalize(self.centroids.data.to(torch.float32), p=2, dim=1).to(self._torch_dtype)
            
        return loss.item()

    def _evaluate_metric(self, X_val: torch.Tensor, y_val: torch.Tensor, sample_weight: torch.Tensor, feature_mask: Optional[torch.Tensor]) -> Tuple[float, bool]:
        with torch.no_grad(): 
            val_loss = F.mse_loss(self.forward(X_val, feature_mask).to(torch.float32), y_val.to(torch.float32), reduction='none').mean(dim=1)
            metric = -torch.sqrt((val_loss * sample_weight.flatten()).mean())
            return metric.item(), True

    def predict(self, X: Any, feature_mask: Any = None) -> torch.Tensor:
        """Predicts the continuous vector targets based on localized IDW routing."""
        with torch.no_grad(): 
            X_t = self._format_input(X)
            fm = self._format_feature_mask(feature_mask, X_t.shape[0], X_t.shape[1])
            return self.forward(X_t, fm)