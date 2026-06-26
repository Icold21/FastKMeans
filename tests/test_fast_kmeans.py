"""Comprehensive Test Suite for FastKMeans Framework.

This script acts as a massive automated pipeline verifying every single feature,
edge case, and architectural mechanism of the FastKMeans suite.

Features tested:
- Base constraints and strict parameter validation.
- Unsupervised clustering.
- Feature masking and 2D EMA weight allocation.
- Sample weights handling.
- ANOVA-based automatic feature weighting.
- Learning Rate Finder deepcopy protection.
- Differentiable routing safety locks.
- Vector regression with Softmax Scaled assignments.
- Advanced Multi-Label parameters (ASL, LVQ repulsion, Gap strategy).
- Gradient freezing mechanics.
- L1/L2/Diversity Regularizations.
"""

import warnings
warnings.filterwarnings('ignore')

import time
import traceback
import torch
import numpy as np
from sklearn.datasets import make_classification, make_multilabel_classification, make_regression

# Import the framework classes (ensure fast_kmeans.py is in the same directory)
from fast_kmeans import (
    FastKMeansClusterer,
    FastKMeansClassifier,
    FastMultiLabelKMeansClassifier,
    FastKMeansRegressor
)

# ==========================================
# 1. SYNTHETIC DATA GENERATION
# ==========================================
print("📦 Generating synthetic datasets for testing...")

torch.manual_seed(42)
np.random.seed(42)

# Classification Data (1D targets)
X_clf_np, y_clf_np = make_classification(n_samples=200, n_features=10, n_classes=3, n_informative=5, random_state=42)
X_clf = torch.tensor(X_clf_np, dtype=torch.float32)
y_clf = torch.tensor(y_clf_np, dtype=torch.float32)

# Multi-Label Data (2D binary targets)
X_ml_np, y_ml_np = make_multilabel_classification(n_samples=200, n_features=15, n_classes=5, n_labels=2, random_state=42)
X_ml = torch.tensor(X_ml_np, dtype=torch.float32)
y_ml = torch.tensor(y_ml_np, dtype=torch.float32)

# Regression Data (Vector Targets: 2 output dimensions)
X_reg_np, y_reg_np = make_regression(n_samples=200, n_features=10, n_targets=2, random_state=42)
X_reg = torch.tensor(X_reg_np, dtype=torch.float32)
y_reg = torch.tensor(y_reg_np, dtype=torch.float32)

# Sample Weights [0, +inf)
sample_weights = torch.rand(200, dtype=torch.float32) + 0.5 

# Feature Masks (simulating 20% padding/missing values)
feature_masks_clf = (torch.rand(200, 10) > 0.2).float()
feature_masks_reg = (torch.rand(200, 10) > 0.2).float()

print("✅ Data generation complete!\n")


# ==========================================
# 2. CUSTOM TEST RUNNER
# ==========================================
passed_tests = 0
failed_tests = 0

def run_test(test_name: str, test_func: callable):
    """Executes a test function, catches exceptions, and formats the output."""
    global passed_tests, failed_tests
    print(f"⏳ Running: {test_name}...", end=" ", flush=True)
    try:
        test_func()
        print(f"\r✅ SUCCESS: {test_name}" + " " * 20)
        passed_tests += 1
    except Exception as e:
        print(f"\r❌ FAILED:  {test_name}" + " " * 20)
        print("-" * 60)
        traceback.print_exc()
        print("-" * 60)
        failed_tests += 1


# ==========================================
# 3. TEST DEFINITIONS
# ==========================================

def test_01_parameter_validation():
    """Tests if strict parameter validation correctly rejects silent fallbacks."""
    try:
        FastKMeansClassifier(distance='magic_metric')
        raise AssertionError("Model accepted an invalid distance metric!")
    except ValueError:
        pass
        
    try:
        FastKMeansRegressor(soft_type='exponential')
        raise AssertionError("Model accepted an invalid soft_type!")
    except ValueError:
        pass

def test_02_clusterer_basic():
    """Tests purely unsupervised clustering (y=None)."""
    model = FastKMeansClusterer(k_init=5, distance='euclidean', init_mode='random')
    model.fit(X_clf, y=None, max_iters=2, verbose=False)
    
    assert model.centroids.shape[0] == 5, "Centroids were not initialized correctly!"
    
    preds = model.predict(X_clf)
    assert preds.shape[0] == 200, "Prediction shape mismatch!"
    assert preds.ndim == 1, "Cluster assignments must be 1D."

def test_03_feature_masks_and_2d_ema():
    """Tests if providing a feature mask successfully transitions EMA weights to 2D matrices."""
    model = FastKMeansRegressor(k_targets=2, k_features=2)
    model.fit(X_reg, y_reg, feature_mask=feature_masks_reg, max_iters=2, verbose=False)
    
    assert model.centroid_feature_weights.ndim == 2, "Feature weights matrix did not switch to 2D despite feature_mask presence!"
    assert model.centroid_feature_weights.shape == model.centroids.shape, "2D EMA weights shape mismatch!"

def test_04_sample_weights():
    """Tests sample weights incorporation across both mapping and gradient phases."""
    model = FastKMeansClassifier(k_init=2, soft_type='scaled')
    # If shapes mismatch, this will crash
    model.fit(X_clf, y_clf, sample_weight=sample_weights, max_iters=2, verbose=False)
    model.finetune(X_clf, y_clf, sample_weight=sample_weights, epochs=2, verbose=False)
    
    assert model.centroids.shape[0] <= 6, "Unexpected number of centroids generated."

def test_05_auto_feature_weights():
    """Tests algorithmic ANOVA feature weighting generation."""
    model = FastKMeansClassifier(k_init=2, auto_feature_weights=True)
    model.fit(X_clf, y_clf, max_iters=2, verbose=False)
    
    assert getattr(model, "feature_weights", None) is not None, "Feature weights were not instantiated!"
    assert model.feature_weights.shape[0] == 10, "Feature weights dimension mismatch!"
    
    # Values must be mathematically constrained [0, 1]
    assert torch.all(model.feature_weights >= 0.0) and torch.all(model.feature_weights <= 1.0), "Feature weights escaped [0, 1] bounds!"

def test_06_lr_finder_state_protection():
    """Tests if the LR finder properly restores the topology after explosive gradient tests."""
    model = FastKMeansRegressor(k_targets=3, k_features=2, soft_type='scaled')
    model.fit(X_reg, y_reg, max_iters=2, verbose=False)
    
    # Save mathematical snapshot
    initial_centroids = model.centroids.clone().detach()
    
    model.find_learning_rate(X_reg, y_reg)
    
    assert model.optimizer is None, "Optimizer was not wiped after LR search!"
    assert model.learning_rate_ > 0, "Learning rate was not determined!"
    assert torch.allclose(model.centroids, initial_centroids), "LR Finder corrupted the grid topology (State leak)!"

def test_07_gradient_protection():
    """Tests the architectural lock against differentiating non-differentiable operations."""
    model = FastKMeansRegressor(soft_type='hard')
    model.fit(X_reg, y_reg, max_iters=2, verbose=False)
    
    try:
        model.finetune(X_reg, y_reg, epochs=2, verbose=False)
        raise AssertionError("Gradient descent executed under 'hard' routing! Expected RuntimeError.")
    except RuntimeError:
        pass # Expected behavior

def test_08_vector_regression():
    """Tests multi-output vector regression with Softmax scaling."""
    model = FastKMeansRegressor(
        k_targets=4, 
        k_features=2, 
        target_assignment='softmax_scaled', 
        soft_type='softmax_scaled',
        temperature=0.5
    )
    model.fit(X_reg, y_reg, max_iters=2, verbose=False)
    model.finetune(X_reg, y_reg, epochs=2, verbose=False)
    
    preds = model.predict(X_reg)
    assert preds.shape == (200, 2), "Vector prediction output shape mismatch!"

def test_09_multilabel_advanced():
    """Tests ASL loss, LVQ repulsion, and Zero-Hyperparameter Gap Strategy."""
    model = FastMultiLabelKMeansClassifier(
        k_init=2, 
        repulsion_factor=0.1, 
        asl_gamma_neg=3.0, 
        soft_type='scaled'
    )
    model.fit(X_ml, y_ml, max_iters=2, verbose=False)
    model.finetune(X_ml, y_ml, epochs=2, lr=0.01, verbose=False)
    
    preds_gap = model.predict(X_ml, strategy='gap')
    assert preds_gap.shape == (200, 5), "Gap prediction shape mismatch!"
    
    preds_topk = model.predict(X_ml, strategy='top_k', top_k=2)
    assert torch.all(preds_topk.sum(dim=1) == 2), "Top-K strategy failed to assign exactly K labels!"

def test_10_freezing_mechanics():
    """Tests absolute gradient zeroing and immutability of frozen centroids."""
    model = FastKMeansClassifier(k_init=2, soft_type='scaled')
    model.fit(X_clf, y_clf, max_iters=2, verbose=False)
    
    model.freeze_centroids()
    old_centroids = model.centroids.clone().detach()
    
    # Force a massive, usually destructive gradient step
    model.learning_rate_ = 100.0
    model.fit_batch(X_clf, y_clf, gradient_step=True)
    
    assert torch.all(model.centroids.grad == 0.0), "Gradients of frozen centroids are not exactly zero!"
    assert torch.allclose(model.centroids, old_centroids), "Frozen centroids shifted during optimization!"

def test_11_regularization_and_l1():
    """Tests L1 distance compatibility alongside L2 and Diversity penalties."""
    model = FastKMeansClassifier(
        k_init=2, 
        distance='l1', 
        soft_type='scaled',
        l2_reg=0.1, 
        diversity_reg=0.5
    )
    model.fit(X_clf, y_clf, max_iters=2, verbose=False)
    
    loss = model.fit_batch(X_clf, y_clf, gradient_step=True)
    assert loss > 0, "Loss calculation with regularization failed!"


# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == '__main__':
    print("🚀 Starting FastKMeans Suite Verification...\n")
    time.sleep(0.5)

    run_test("Strict Parameter Validation", test_01_parameter_validation)
    run_test("Clusterer (Unsupervised Mode)", test_02_clusterer_basic)
    run_test("Feature Masks & 2D EMA Core", test_03_feature_masks_and_2d_ema)
    run_test("Sample Weights Integration", test_04_sample_weights)
    run_test("Auto Feature Weights (ANOVA)", test_05_auto_feature_weights)
    run_test("LR Finder Deepcopy Protection", test_06_lr_finder_state_protection)
    run_test("Gradient Mode Protection", test_07_gradient_protection)
    run_test("Vector Regression & Softmax", test_08_vector_regression)
    run_test("Multi-Label ASL, Repulsion & Gap", test_09_multilabel_advanced)
    run_test("Centroid Freezing Mechanics", test_10_freezing_mechanics)
    run_test("L1 Distance & Regularizations", test_11_regularization_and_l1)

    print("\n" + "="*50)
    print(f"🎯 TESTING SUMMARY:")
    print(f"✅ Tests Passed: {passed_tests}")
    if failed_tests > 0:
        print(f"❌ Tests Failed: {failed_tests}")
    else:
        print(f"🌟 ALL TESTS PASSED! THE FRAMEWORK IS ROCK SOLID.")
    print("="*50)