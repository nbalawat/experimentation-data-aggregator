from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import numpy as np
from datetime import datetime
import pandas as pd
from scipy import stats
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentType(Enum):
    AB_TEST = "ab_test"
    MULTI_VARIATE = "multivariate"
    BANDIT = "bandit"

@dataclass
class ExperimentConfig:
    """Configuration for experiment setup"""
    name: str
    type: ExperimentType
    variants: List[str]
    metrics: List[str]
    duration_days: int
    confidence_level: float = 0.95
    minimum_sample_size: int = 100

class CustomerSegment:
    """Handle customer segmentation for experiments"""
    def __init__(self, segment_criteria: Dict[str, Any]):
        self.criteria = segment_criteria
    
    def filter_customers(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """Filter customers based on segmentation criteria"""
        filtered_df = customers_df.copy()
        for column, value in self.criteria.items():
            filtered_df = filtered_df[filtered_df[column] == value]
        return filtered_df

class ExperimentGroup(ABC):
    """Abstract base class for different types of experiment groups"""
    @abstractmethod
    def assign_variants(self, customer_ids: List[str]) -> Dict[str, str]:
        pass
    
    @abstractmethod
    def calculate_results(self, metrics_data: pd.DataFrame) -> Dict[str, Any]:
        pass

class ABTestGroup(ExperimentGroup):
    """Implementation for A/B testing"""
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
    def assign_variants(self, customer_ids: List[str]) -> Dict[str, str]:
        """Randomly assign customers to variants"""
        np.random.seed(42)  # For reproducibility
        assignments = {}
        for customer_id in customer_ids:
            variant = np.random.choice(self.config.variants)
            assignments[customer_id] = variant
        return assignments
    
    def calculate_results(self, metrics_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistical significance between variants"""
        results = {}
        for metric in self.config.metrics:
            control_data = metrics_data[metrics_data['variant'] == self.config.variants[0]][metric]
            treatment_data = metrics_data[metrics_data['variant'] == self.config.variants[1]][metric]
            
            t_stat, p_value = stats.ttest_ind(control_data, treatment_data)
            effect_size = (treatment_data.mean() - control_data.mean()) / control_data.mean()
            
            results[metric] = {
                'p_value': p_value,
                'significant': p_value < (1 - self.config.confidence_level),
                'effect_size': effect_size,
                'control_mean': control_data.mean(),
                'treatment_mean': treatment_data.mean()
            }
        return results

class LoyaltyExperiment:
    """Main experiment orchestrator"""
    def __init__(
        self,
        config: ExperimentConfig,
        segment: Optional[CustomerSegment] = None
    ):
        self.config = config
        self.segment = segment
        self.experiment_group = self._create_experiment_group()
        self.start_time = None
        self.assignments = {}
        
    def _create_experiment_group(self) -> ExperimentGroup:
        """Factory method to create appropriate experiment group"""
        if self.config.type == ExperimentType.AB_TEST:
            return ABTestGroup(self.config)
        # Add other experiment types as needed
        raise ValueError(f"Unsupported experiment type: {self.config.type}")
    
    def start_experiment(self, customers_df: pd.DataFrame):
        """Initialize and start the experiment"""
        if self.segment:
            customers_df = self.segment.filter_customers(customers_df)
            
        if len(customers_df) < self.config.minimum_sample_size:
            raise ValueError(f"Insufficient sample size: {len(customers_df)} < {self.config.minimum_sample_size}")
        
        self.start_time = datetime.now()
        self.assignments = self.experiment_group.assign_variants(customers_df['customer_id'].tolist())
        logger.info(f"Started experiment: {self.config.name} with {len(self.assignments)} customers")
        
    def get_variant(self, customer_id: str) -> str:
        """Get assigned variant for a customer"""
        return self.assignments.get(customer_id)
    
    def analyze_results(self, metrics_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze experiment results"""
        if not self.start_time:
            raise ValueError("Experiment hasn't started yet")
            
        results = self.experiment_group.calculate_results(metrics_data)
        logger.info(f"Analysis completed for experiment: {self.config.name}")
        return results

# Example usage
def run_loyalty_experiment():
    # Configure experiment
    config = ExperimentConfig(
        name="premium_rewards_test",
        type=ExperimentType.AB_TEST,
        variants=["control", "premium_rewards"],
        metrics=["purchase_amount", "visit_frequency"],
        duration_days=30,
        confidence_level=0.95
    )
    
    # Define customer segment
    segment = CustomerSegment({
        'tier': 'gold',
        'active_status': True
    })
    
    # Initialize experiment
    experiment = LoyaltyExperiment(config, segment)
    
    # Sample data for demonstration
    customers_df = pd.DataFrame({
        'customer_id': [f'cust_{i}' for i in range(1000)],
        'tier': np.random.choice(['gold', 'silver'], size=1000),
        'active_status': np.random.choice([True, False], size=1000)
    })
    
    # Start experiment
    experiment.start_experiment(customers_df)
    
    # Simulate collecting metrics
    metrics_data = pd.DataFrame({
        'customer_id': list(experiment.assignments.keys()),
        'variant': list(experiment.assignments.values()),
        'purchase_amount': np.random.normal(100, 20, len(experiment.assignments)),
        'visit_frequency': np.random.poisson(5, len(experiment.assignments))
    })
    
    # Analyze results
    results = experiment.analyze_results(metrics_data)
    return results

if __name__ == "__main__":
    results = run_loyalty_experiment()
    print("Experiment Results:", results)