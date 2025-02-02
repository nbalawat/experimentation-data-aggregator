import pytest
import numpy as np
import polars as pl
import os
import sys
from pathlib import Path

# Ensure framework is in path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from framework.file3 import (
    EnhancedLoyaltyExperiment,
    AdvancedExperimentConfig,
    ExperimentType
)

@pytest.fixture
def base_config():
    return AdvancedExperimentConfig(
        name="test_experiment",
        type=ExperimentType.AB_TEST,
        variants=["control", "premium_rewards", "premium_plus"],
        metrics=["purchase_amount", "visit_frequency", "satisfaction_score"],
        duration_days=30,
        confidence_level=0.95,
        persistence_path="test_data",
        minimum_sample_size=100,
        cross_validation_folds=5
    )

def generate_test_data(size: int, control_mean: float, treatment_means: dict, std: float = 1.0) -> pl.DataFrame:
    """Generate synthetic test data with specified means and standard deviation."""
    np.random.seed(42)  # For reproducibility
    
    data = []
    variants = ["control"] + list(treatment_means.keys())
    customers_per_variant = size // len(variants)
    
    for variant in variants:
        mean = control_mean if variant == "control" else treatment_means[variant]
        values = np.random.normal(mean, std, customers_per_variant)
        
        data.extend([
            {
                "customer_id": f"cust_{i}",
                "variant": variant,
                "purchase_amount": values[i],
                "visit_frequency": np.random.poisson(values[i] / 20),  # Related to purchase amount
                "satisfaction_score": min(10, max(1, np.random.normal(values[i] / 10, 0.5)))  # Scale to 1-10
            }
            for i in range(customers_per_variant)
        ])
    
    return pl.DataFrame(data)

def test_successful_experiment(base_config):
    """Test case where both variants show significant improvements."""
    # Generate data where treatments perform better
    data = generate_test_data(
        size=900,  # 300 per variant
        control_mean=100,
        treatment_means={
            "premium_rewards": 120,  # 20% improvement
            "premium_plus": 115      # 15% improvement
        },
        std=20
    )
    
    experiment = EnhancedLoyaltyExperiment(base_config)
    results = experiment.analyze_results(data)
    
    # Verify success conditions
    assert results["overall_success"]
    assert results["purchase_amount_success"]["premium_rewards"]
    assert results["purchase_amount_success"]["premium_plus"]
    
    # Verify effect sizes and p-values
    for variant in ["premium_rewards", "premium_plus"]:
        stats = results["purchase_amount_stats"][variant]
        assert stats["effect_size"] > 0.2
        assert stats["p_value"] < 0.05
        assert stats["lift_percentage"] > 0

def test_failed_experiment(base_config):
    """Test case where variants show no significant improvement."""
    # Generate data where treatments perform similarly to control
    data = generate_test_data(
        size=900,
        control_mean=100,
        treatment_means={
            "premium_rewards": 101,  # 1% improvement (not significant)
            "premium_plus": 99       # -1% decline (not significant)
        },
        std=20
    )
    
    experiment = EnhancedLoyaltyExperiment(base_config)
    results = experiment.analyze_results(data)
    
    # Verify failure conditions
    assert not results["overall_success"]
    assert not results["purchase_amount_success"]["premium_rewards"]
    assert not results["purchase_amount_success"]["premium_plus"]
    
    # Verify effect sizes and p-values indicate no significant difference
    for variant in ["premium_rewards", "premium_plus"]:
        stats = results["purchase_amount_stats"][variant]
        assert abs(stats["effect_size"]) < 0.2
        assert stats["p_value"] > 0.05

def test_mixed_results_experiment(base_config):
    """Test case where one variant improves and another degrades."""
    # Generate data where one treatment improves and another degrades
    data = generate_test_data(
        size=900,
        control_mean=100,
        treatment_means={
            "premium_rewards": 120,  # 20% improvement
            "premium_plus": 85       # 15% degradation
        },
        std=20
    )
    
    experiment = EnhancedLoyaltyExperiment(base_config)
    results = experiment.analyze_results(data)
    
    # Verify mixed results conditions
    assert not results["overall_success"]  # Should fail due to significant degradation
    assert results["purchase_amount_success"]["premium_rewards"]
    assert not results["purchase_amount_success"]["premium_plus"]
    
    # Verify premium_rewards shows improvement
    premium_rewards_stats = results["purchase_amount_stats"]["premium_rewards"]
    assert premium_rewards_stats["effect_size"] > 0.2
    assert premium_rewards_stats["p_value"] < 0.05
    assert premium_rewards_stats["lift_percentage"] > 0
    
    # Verify premium_plus shows degradation
    premium_plus_stats = results["purchase_amount_stats"]["premium_plus"]
    assert premium_plus_stats["effect_size"] < -0.2
    assert premium_plus_stats["p_value"] < 0.05
    assert premium_plus_stats["lift_percentage"] < 0

def test_small_sample_size(base_config):
    """Test case with small sample size that shouldn't show significance."""
    # Generate data with small sample size
    data = generate_test_data(
        size=30,  # Only 10 per variant
        control_mean=100,
        treatment_means={
            "premium_rewards": 120,  # 20% improvement
            "premium_plus": 115      # 15% improvement
        },
        std=20
    )
    
    experiment = EnhancedLoyaltyExperiment(base_config)
    results = experiment.analyze_results(data)
    
    # Verify that even with large effects, small sample size prevents significance
    assert not results["overall_success"]
    for variant in ["premium_rewards", "premium_plus"]:
        stats = results["purchase_amount_stats"][variant]
        assert stats["p_value"] > 0.05  # Should not be significant due to small sample
