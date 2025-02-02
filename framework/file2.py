from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np
from datetime import datetime
import pandas as pd
from scipy import stats
import logging
from abc import ABC, abstractmethod
import sqlite3
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency, power_analysis
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentType(Enum):
    AB_TEST = "ab_test"
    MULTI_VARIATE = "multivariate"
    BANDIT = "bandit"
    FACTORIAL = "factorial"

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
    persistence_path: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for storage"""
        return {
            'name': self.name,
            'type': self.type.value,
            'variants': self.variants,
            'metrics': self.metrics,
            'duration_days': self.duration_days,
            'confidence_level': self.confidence_level,
            'minimum_sample_size': self.minimum_sample_size
        }

class DataPersistence:
    """Handle experiment data persistence"""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self):
        """Create necessary tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    config TEXT,
                    start_time TEXT,
                    status TEXT
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS assignments (
                    experiment_id INTEGER,
                    customer_id TEXT,
                    variant TEXT,
                    assignment_time TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    experiment_id INTEGER,
                    customer_id TEXT,
                    metric_name TEXT,
                    value REAL,
                    timestamp TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            ''')

    def save_experiment(self, config: ExperimentConfig, start_time: datetime) -> int:
        """Save experiment configuration and return experiment ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'INSERT INTO experiments (name, config, start_time, status) VALUES (?, ?, ?, ?)',
                (config.name, json.dumps(config.to_dict()), start_time.isoformat(), 'RUNNING')
            )
            return cursor.lastrowid

    def save_assignments(self, experiment_id: int, assignments: Dict[str, str]):
        """Save variant assignments"""
        with sqlite3.connect(self.db_path) as conn:
            assignment_time = datetime.now().isoformat()
            conn.executemany(
                'INSERT INTO assignments VALUES (?, ?, ?, ?)',
                [(experiment_id, customer_id, variant, assignment_time) 
                 for customer_id, variant in assignments.items()]
            )

    def save_metrics(self, experiment_id: int, metrics_data: pd.DataFrame):
        """Save metrics data"""
        with sqlite3.connect(self.db_path) as conn:
            for metric in metrics_data.columns:
                if metric not in ['customer_id', 'variant']:
                    timestamp = datetime.now().isoformat()
                    data = [(experiment_id, row['customer_id'], metric, row[metric], timestamp)
                           for _, row in metrics_data.iterrows()]
                    conn.executemany(
                        'INSERT INTO metrics VALUES (?, ?, ?, ?, ?)',
                        data
                    )

class PowerAnalysis:
    """Handle statistical power analysis"""
    @staticmethod
    def calculate_required_sample_size(
        effect_size: float,
        power: float = 0.8,
        alpha: float = 0.05
    ) -> int:
        """Calculate required sample size for desired statistical power"""
        analysis = power_analysis.TTestIndPower()
        n = analysis.solve_power(
            effect_size=effect_size,
            power=power,
            alpha=alpha
        )
        return int(np.ceil(n))

class MultiVariateTestGroup(ExperimentGroup):
    """Implementation for multivariate testing"""
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
    def assign_variants(self, customer_ids: List[str]) -> Dict[str, str]:
        """Assign customers to variant combinations"""
        np.random.seed(42)
        variant_combinations = self._generate_variant_combinations()
        assignments = {}
        for customer_id in customer_ids:
            variant = np.random.choice(variant_combinations)
            assignments[customer_id] = variant
        return assignments
    
    def _generate_variant_combinations(self) -> List[str]:
        """Generate all possible variant combinations"""
        # Implementation depends on variant structure
        return self.config.variants
    
    def calculate_results(self, metrics_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistical significance for multivariate test"""
        results = {}
        for metric in self.config.metrics:
            # Perform ANOVA test
            variants = metrics_data['variant'].unique()
            variant_data = [metrics_data[metrics_data['variant'] == variant][metric] 
                          for variant in variants]
            f_stat, p_value = stats.f_oneway(*variant_data)
            
            # Calculate effect sizes
            total_mean = metrics_data[metric].mean()
            variant_means = {variant: metrics_data[metrics_data['variant'] == variant][metric].mean() 
                           for variant in variants}
            effect_sizes = {variant: (mean - total_mean) / total_mean 
                          for variant, mean in variant_means.items()}
            
            results[metric] = {
                'p_value': p_value,
                'significant': p_value < (1 - self.config.confidence_level),
                'f_statistic': f_stat,
                'effect_sizes': effect_sizes,
                'variant_means': variant_means
            }
        return results

class BanditTestGroup(ExperimentGroup):
    """Implementation for multi-armed bandit testing"""
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.variant_stats = {variant: {'trials': 0, 'successes': 0} 
                            for variant in config.variants}
    
    def assign_variants(self, customer_ids: List[str]) -> Dict[str, str]:
        """Assign variants using Thompson sampling"""
        assignments = {}
        for customer_id in customer_ids:
            variant = self._thompson_sampling()
            assignments[customer_id] = variant
        return assignments
    
    def _thompson_sampling(self) -> str:
        """Implement Thompson sampling for variant selection"""
        samples = {}
        for variant, stats in self.variant_stats.items():
            alpha = stats['successes'] + 1
            beta = stats['trials'] - stats['successes'] + 1
            samples[variant] = np.random.beta(alpha, beta)
        return max(samples, key=samples.get)
    
    def calculate_results(self, metrics_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate bandit test results"""
        results = {}
        for metric in self.config.metrics:
            variant_stats = {}
            for variant in self.config.variants:
                variant_data = metrics_data[metrics_data['variant'] == variant][metric]
                variant_stats[variant] = {
                    'mean': variant_data.mean(),
                    'std': variant_data.std(),
                    'count': len(variant_data),
                    'total': variant_data.sum()
                }
            
            results[metric] = {
                'variant_stats': variant_stats,
                'best_variant': max(variant_stats.items(), 
                                  key=lambda x: x[1]['mean'])[0]
            }
        return results

class LoyaltyExperiment:
    """Main experiment orchestrator with enhanced capabilities"""
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
        self.persistence = (DataPersistence(config.persistence_path) 
                          if config.persistence_path else None)
        self.experiment_id = None
        
    def _create_experiment_group(self) -> ExperimentGroup:
        """Factory method to create appropriate experiment group"""
        if self.config.type == ExperimentType.AB_TEST:
            return ABTestGroup(self.config)
        elif self.config.type == ExperimentType.MULTI_VARIATE:
            return MultiVariateTestGroup(self.config)
        elif self.config.type == ExperimentType.BANDIT:
            return BanditTestGroup(self.config)
        raise ValueError(f"Unsupported experiment type: {self.config.type}")
    
    def start_experiment(self, customers_df: pd.DataFrame):
        """Initialize and start the experiment with persistence"""
        if self.segment:
            customers_df = self.segment.filter_customers(customers_df)
            
        if len(customers_df) < self.config.minimum_sample_size:
            raise ValueError(f"Insufficient sample size: {len(customers_df)} < {self.config.minimum_sample_size}")
        
        self.start_time = datetime.now()
        self.assignments = self.experiment_group.assign_variants(customers_df['customer_id'].tolist())
        
        if self.persistence:
            self.experiment_id = self.persistence.save_experiment(self.config, self.start_time)
            self.persistence.save_assignments(self.experiment_id, self.assignments)
            
        logger.info(f"Started experiment: {self.config.name} with {len(self.assignments)} customers")
    
    def record_metrics(self, metrics_data: pd.DataFrame):
        """Record metrics with persistence"""
        if self.persistence and self.experiment_id:
            self.persistence.save_metrics(self.experiment_id, metrics_data)
    
    def analyze_results(self, metrics_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze experiment results with enhanced statistics"""
        if not self.start_time:
            raise ValueError("Experiment hasn't started yet")
            
        results = self.experiment_group.calculate_results(metrics_data)
        
        # Add advanced analytics
        for metric in self.config.metrics:
            results[metric]['summary_stats'] = {
                'overall_mean': metrics_data[metric].mean(),
                'overall_std': metrics_data[metric].std(),
                'quartiles': metrics_data[metric].quantile([0.25, 0.5, 0.75]).to_dict()
            }
            
            # Add confidence intervals
            for variant in self.config.variants:
                variant_data = metrics_data[metrics_data['variant'] == variant][metric]
                ci = stats.t.interval(
                    self.config.confidence_level,
                    len(variant_data) - 1,
                    variant_data.mean(),
                    stats.sem(variant_data)
                )
                results[metric][f'{variant}_ci'] = ci
        
        logger.info(f"Analysis completed for experiment: {self.config.name}")
        return results

# Example usage with enhanced features
def run_advanced_loyalty_experiment():
    # Configure experiment with persistence
    config = ExperimentConfig(
        name="premium_rewards_test",
        type=ExperimentType.MULTI_VARIATE,
        variants=["control", "premium_rewards", "premium_plus"],
        metrics=["purchase_amount", "visit_frequency", "satisfaction_score"],
        duration_days=30,
        confidence_level=0.95,
        persistence_path="loyalty_experiments.db"
    )
    
    # Define customer segment
    segment = CustomerSegment({
        'tier': 'gold',
        'active_status': True
    })
    
    # Calculate required sample size
    required_size = PowerAnalysis.calculate_required_sample_size(
        effect_size=0.3,  # Medium effect size
        power=0.8,
        alpha=0.05
    )
    
    # Initialize experiment
    experiment = LoyaltyExperiment(config, segment)
    
    # Generate sample data
    customers_df = pd.DataFrame({
        'customer_id': [f'cust_{i}' for i in range(max(1000, required_size))],
        'tier': np.random.choice(['gold', 'silver'], size=max(1000, required_size)),
        'active_status': np.random.choice([True, False], size=max(1000, required_size))
    })
    
    # Start experiment
    experiment.start_experiment(customers_df)
    
    # Simulate collecting metrics
    metrics_data = pd.DataFrame({
        'customer_id': list(experiment.assignments.keys()),
        'variant': list(experiment.assignments.values()),
        'purchase_amount': np.random.normal(100, 20, len(experiment.assignments)),
        'visit_frequency': np.random.poisson(5, len(experiment.assignments)),
        'satisfaction_score': np.random.normal(8, 1, len(experiment.assignments))
    })
    
    # Record metrics
    experiment.record_metrics(metrics_data)
    
    # Analyze results
    results = experiment.analyze_results(metrics_data)
    return results

if __name__ == "__main__":
    results = run_advanced_loyalty_experiment()
    print("Experiment Results:", results)