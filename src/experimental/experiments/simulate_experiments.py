from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
import logging
import numpy as np
import polars as pl
from pathlib import Path
import warnings
from statsmodels.stats.multitest import multipletests
from scipy.stats import beta
from experimental.experiments.base import BaseExperiment, ExperimentGroup
from datetime import datetime
import sqlite3
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency, norm
from statsmodels.stats import power
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import json
import scipy.stats as stats
from tabulate import tabulate

# Configure logging with a more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)

class BanditStrategy(str, Enum):
    """Available strategies for bandit experiments."""
    THOMPSON = "thompson"
    UCB = "ucb"
    EPSILON_GREEDY = "epsilon_greedy"

class SequentialTestType(str, Enum):
    """Types of sequential testing procedures."""
    OBRIEN_FLEMING = "obrien_fleming"
    POCOCK = "pocock"
    HAYBITTLE_PETO = "haybittle_peto"

class ExperimentType(str, Enum):
    """Types of experiments supported."""
    AB_TEST = "ab_test"
    BANDIT = "bandit"
    MULTIVARIATE = "multivariate"

@dataclass
class AdvancedExperimentConfig:
    """Enhanced configuration with advanced features"""
    name: str
    type: ExperimentType
    variants: List[str]
    metrics: List[str]
    duration_days: int
    confidence_level: float
    persistence_path: str
    minimum_sample_size: int
    visualization_path: Optional[str] = None
    bandit_strategy: Optional[BanditStrategy] = None
    sequential_test: Optional[SequentialTestType] = None
    cross_validation_folds: int = 5
    epsilon: float = 0.1  # For epsilon-greedy
    ucb_alpha: float = 1.0  # For UCB

class SequentialTesting:
    """Handle sequential testing with multiple stopping points."""
    def __init__(self, test_type: SequentialTestType, total_looks: int):
        self.test_type = test_type
        self.total_looks = total_looks
        self.current_look = 0
        logger.info(
            f"Initializing sequential testing with type: {test_type.value}, "
            f"total looks: {total_looks}"
        )
    
    def get_adjusted_alpha(self) -> float:
        """Get adjusted significance level based on sequential test type."""
        self.current_look += 1
        logger.debug(
            f"Calculating adjusted alpha for look {self.current_look}/{self.total_looks}"
        )
        
        if self.test_type == SequentialTestType.OBRIEN_FLEMING:
            alpha = self._obrien_fleming_boundary()
            logger.info(f"O'Brien-Fleming boundary for look {self.current_look}: {alpha:.4f}")
            return alpha
        elif self.test_type == SequentialTestType.POCOCK:
            alpha = self._pocock_boundary()
            logger.info(f"Pocock boundary for look {self.current_look}: {alpha:.4f}")
            return alpha
        else:  # Haybittle-Peto
            alpha = 0.001 if self.current_look < self.total_looks else 0.05
            logger.info(f"Haybittle-Peto boundary for look {self.current_look}: {alpha:.4f}")
            return alpha

    def _obrien_fleming_boundary(self) -> float:
        """Calculate O'Brien-Fleming boundary."""
        logger.debug("Calculating O'Brien-Fleming boundary")
        return np.sqrt(2 * np.log(np.log(self.total_looks))) * np.sqrt(1 / self.total_looks)

    def _pocock_boundary(self) -> float:
        """Calculate Pocock boundary."""
        logger.debug("Calculating Pocock boundary")
        return 0.0221 * np.log(1 + (np.e - 1) * self.current_look / self.total_looks)

class CrossValidation:
    """Handle cross-validation for experiment results."""
    
    def __init__(self, n_splits: int, stratify: bool = True):
        self.n_splits = n_splits
        self.stratify = stratify
        logger.info(
            f"Initializing {n_splits}-fold cross-validation "
            f"{'with' if stratify else 'without'} stratification"
        )
    
    def split_data(self, data: pl.DataFrame, metric: str) -> List[Tuple[pl.DataFrame, pl.DataFrame]]:
        """Split data for cross-validation.
        
        Args:
            data: DataFrame containing the data
            metric: Name of the metric column to analyze
            
        Returns:
            List of (train_data, test_data) tuples for each fold
        """
        logger.info(f"Splitting data with {data.height} samples into {self.n_splits} folds")
        
        if self.stratify:
            # Stratify by variant to maintain treatment group proportions
            logger.debug("Using stratified split on variant column")
            if 'variant' not in data.columns:
                logger.warning("No variant column found for stratification, falling back to regular k-fold")
                self.stratify = False
            else:
                variants = data.select(pl.col('variant')).to_numpy().flatten()
                kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
                splits = kf.split(np.zeros(len(variants)), variants)
        
        if not self.stratify:
            logger.debug("Using standard k-fold split")
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            splits = kf.split(np.zeros(data.height))
        
        result = []
        for fold, (train_idx, test_idx) in enumerate(splits):
            # Create boolean masks for polars filtering
            train_mask = pl.Series("_idx", [i in train_idx for i in range(data.height)])
            test_mask = pl.Series("_idx", [i in test_idx for i in range(data.height)])
            
            # Use lazy operations to prevent printing
            train_data = data.lazy().filter(train_mask).collect(streaming=True)
            test_data = data.lazy().filter(test_mask).collect(streaming=True)
            
            # Log variant distribution in train and test sets if stratifying
            if self.stratify:
                train_dist = train_data.group_by('variant').agg(pl.len()).sort('variant')
                test_dist = test_data.group_by('variant').agg(pl.len()).sort('variant')
                logger.debug(
                    f"Fold {fold} variant distribution:\n"
                    f"Train: {dict(zip(train_dist['variant'], train_dist['len']))}\n"
                    f"Test: {dict(zip(test_dist['variant'], test_dist['len']))}"
                )
            
            logger.debug(
                f"Fold {fold}: train_size={train_data.height}, "
                f"test_size={test_data.height}"
            )
            result.append((train_data, test_data))
        
        return result

class VisualizationManager:
    """Handle experiment result visualizations."""
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initializing visualization manager with output path: {output_path}")
    
    def plot_metric_distributions(self, metrics_data: pl.DataFrame, metric: str):
        """Plot distribution of metric across variants."""
        logger.info(f"Plotting distribution for metric: {metric}")
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='variant', y=metric, data=metrics_data.to_pandas())
        plt.title(f'Distribution of {metric} by Variant')
        output_file = self.output_path / f'{metric}_distribution.png'
        plt.savefig(output_file)
        plt.close()
        logger.info(f"Saved distribution plot to: {output_file}")
    
    def plot_sequential_boundaries(self, sequential_test: SequentialTesting):
        """Plot sequential testing boundaries."""
        logger.info("Plotting sequential testing boundaries")
        looks = range(1, sequential_test.total_looks + 1)
        boundaries = [sequential_test.get_adjusted_alpha() for _ in looks]
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(looks, boundaries, marker='o')
        plt.title('Sequential Testing Boundaries')
        plt.xlabel('Look Number')
        plt.ylabel('Adjusted Alpha')
        output_file = self.output_path / 'sequential_boundaries.png'
        plt.savefig(output_file)
        plt.close()
        logger.info(f"Saved sequential boundaries plot to: {output_file}")

class AdvancedBanditTestGroup(ExperimentGroup):
    """Enhanced bandit testing with multiple strategies."""
    
    def __init__(self, config: AdvancedExperimentConfig):
        super().__init__(config)
        self.strategy = config.bandit_strategy
        self.epsilon = config.epsilon
        self.ucb_alpha = config.ucb_alpha
        self.variants = config.variants
        # Initialize success/failure counts for Thompson sampling
        self.successes = {variant: 1 for variant in config.variants}  # Beta prior a=1
        self.failures = {variant: 1 for variant in config.variants}   # Beta prior b=1
        self.selection_history = {variant: [] for variant in config.variants}
        logger.info(
            f"Initializing advanced bandit test with strategy: {self.strategy}, "
            f"epsilon: {self.epsilon}, ucb_alpha: {self.ucb_alpha}"
        )
    
    def assign_variants(self, customer_ids: List[str]) -> Dict[str, str]:
        """Assign variants using selected strategy."""
        assignments = {}
        logger.info(f"Assigning variants to {len(customer_ids)} customers using {self.strategy} strategy")
        
        for customer_id in customer_ids:
            if self.strategy == BanditStrategy.THOMPSON:
                variant = self._thompson_sampling()
            elif self.strategy == BanditStrategy.UCB:
                variant = self._ucb_selection()
            elif self.strategy == BanditStrategy.EPSILON_GREEDY:
                variant = self._epsilon_greedy()
            else:
                # Default to random assignment if strategy not recognized
                variant = np.random.choice(self.variants)
            
            assignments[customer_id] = variant
            self._update_selection_history(variant)
        
        # Log assignment distribution
        distribution = {v: list(assignments.values()).count(v) for v in self.variants}
        logger.info(f"Variant distribution: {distribution}")
        
        return assignments
    
    def _thompson_sampling(self) -> str:
        """Thompson sampling selection strategy."""
        samples = {
            variant: np.random.beta(
                self.successes[variant],
                self.failures[variant]
            )
            for variant in self.variants
        }
        return max(samples.items(), key=lambda x: x[1])[0]
    
    def _ucb_selection(self) -> str:
        """Upper Confidence Bound selection."""
        total_trials = sum(len(history) for history in self.selection_history.values())
        if total_trials == 0:
            return np.random.choice(self.variants)
        
        ucb_values = {}
        for variant in self.variants:
            trials = len(self.selection_history[variant])
            if trials == 0:
                return variant  # Always try unused variants first
            
            success_rate = self.successes[variant] / (self.successes[variant] + self.failures[variant])
            exploration_term = np.sqrt(
                (2 * np.log(total_trials)) / trials
            ) * self.ucb_alpha
            ucb_values[variant] = success_rate + exploration_term
        
        return max(ucb_values.items(), key=lambda x: x[1])[0]
    
    def _epsilon_greedy(self) -> str:
        """Epsilon-greedy selection."""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.variants)  # Explore
        
        # Exploit: choose variant with highest success rate
        success_rates = {
            variant: self.successes[variant] / (self.successes[variant] + self.failures[variant])
            for variant in self.variants
        }
        return max(success_rates.items(), key=lambda x: x[1])[0]
    
    def _update_selection_history(self, selected_variant: str) -> None:
        """Update selection history for visualization."""
        current_time = datetime.now()
        self.selection_history[selected_variant].append(current_time)
    
    def calculate_results(self, metrics_data: pl.DataFrame) -> Dict[str, Any]:
        """Calculate experiment results."""
        results = {}
        
        for metric in self.config.metrics:
            if metric in metrics_data.columns:
                # Calculate mean and confidence intervals for each variant
                variant_stats = {}
                for variant in self.config.variants:
                    variant_data = metrics_data.filter(pl.col('variant') == variant)
                    if variant_data.height > 0:
                        variant_stats[variant] = {
                            'mean': variant_data.select(pl.col(metric).mean()).item(),
                            'std': variant_data.select(pl.col(metric).std()).item(),
                            'count': variant_data.height
                        }
                
                results[metric] = variant_stats
        
        return results

class EnhancedLoyaltyExperiment(BaseExperiment):
    """Enhanced experiment orchestrator with advanced features."""
    
    def __init__(
            self,
            config: AdvancedExperimentConfig,
            segment: Optional[Dict[str, Any]] = None
        ):
        super().__init__(config)
        self.segment = segment
        self.sequential_test = (
            SequentialTesting(config.sequential_test, 5) 
            if config.sequential_test else None
        )
        self.cross_validation = CrossValidation(config.cross_validation_folds)
        self.visualization = (
            VisualizationManager(config.visualization_path) 
            if config.visualization_path else None
        )
        logger.info("Initialized enhanced loyalty experiment")
    
    def _create_experiment_group(self) -> ExperimentGroup:
        """Create appropriate experiment group based on configuration."""
        if self.config.bandit_strategy:
            return AdvancedBanditTestGroup(self.config)
        raise ValueError("Unsupported experiment type")

    def analyze_results(self, metrics_data: pl.DataFrame) -> Dict[str, Any]:
        """Enhanced analysis with sequential testing and cross-validation."""
        logger.info("Starting enhanced analysis")
        results = {}
        
        # Perform sequential testing if configured
        if self.sequential_test:
            adjusted_alpha = self.sequential_test.get_adjusted_alpha()
            logger.info(f"Using adjusted alpha: {adjusted_alpha}")
            
            # Add sequential testing results
            results['sequential_testing'] = {
                'adjusted_alpha': adjusted_alpha,
                'current_look': self.sequential_test.current_look
            }
            
            # Plot sequential boundaries if visualization is enabled
            if self.visualization:
                self.visualization.plot_sequential_boundaries(self.sequential_test)
        
        # Analyze each metric
        for metric in self.config.metrics:
            if metric not in metrics_data.columns:
                logger.warning(f"Metric {metric} not found in data")
                continue
                
            # Plot metric distributions if visualization is enabled
            if self.visualization:
                self.visualization.plot_metric_distributions(metrics_data, metric)
            
            # Calculate metric statistics
            metric_stats = self._calculate_metric_statistics(metrics_data, metric)
            results[f'{metric}_stats'] = metric_stats
            
            # Determine success for this metric
            adjusted_alpha = (
                self.sequential_test.get_adjusted_alpha()
                if self.sequential_test
                else 0.05
            )
            success = self._is_metric_successful(metric_stats, adjusted_alpha)
            results[f'{metric}_success'] = success
            
            # Cross-validation results
            cv_results = self.cross_validation.split_data(metrics_data, metric)
            # Store only the fold sizes, not the full DataFrames
            results[f'cv_{metric}'] = [
                {'train_size': train.height, 'test_size': test.height}
                for train, test in cv_results
            ]
            logger.debug(f"Cross-validation completed for {metric} with {len(cv_results)} folds")
        
        # Overall experiment success
        results['overall_success'] = self._determine_overall_success(results)
        
        # Display results in table format
        self._display_results_table(results)
        
        return results
    
    def _calculate_metric_statistics(self, data: pl.DataFrame, metric: str) -> Dict[str, Any]:
        """Calculate key statistics for a metric."""
        control_data = data.filter(pl.col('variant') == 'control').select(pl.col(metric)).to_numpy().flatten()
        
        # Calculate statistics for each treatment variant
        variant_stats = {}
        for variant in self.config.variants:
            if variant == 'control':
                continue
                
            variant_data = data.filter(pl.col('variant') == variant).select(pl.col(metric)).to_numpy().flatten()
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(control_data) + np.var(variant_data)) / 2)
            effect_size = (np.mean(variant_data) - np.mean(control_data)) / pooled_std if pooled_std != 0 else 0
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(variant_data, control_data)
            
            variant_stats[variant] = {
                'effect_size': effect_size,
                'p_value': p_value,
                't_statistic': t_stat,
                'control_mean': np.mean(control_data),
                'treatment_mean': np.mean(variant_data),
                'control_std': np.std(control_data),
                'treatment_std': np.std(variant_data),
                'lift_percentage': ((np.mean(variant_data) - np.mean(control_data)) / np.mean(control_data)) * 100
            }
            
        return variant_stats
    
    def _is_metric_successful(self, variant_stats: Dict[str, Dict[str, float]], alpha: float) -> Dict[str, bool]:
        """Determine if a metric shows significant improvement for each variant."""
        success_by_variant = {}
        for variant, stats in variant_stats.items():
            # A metric is successful if:
            # 1. It's statistically significant (p < alpha)
            # 2. Has meaningful positive effect size (> 0.2)
            success_by_variant[variant] = (
                stats['p_value'] < alpha and
                stats['effect_size'] > 0.2
            )
        return success_by_variant

    def _display_results_table(self, results: Dict[str, Any]) -> None:
        """Display experiment results in a formatted table."""
        from tabulate import tabulate
        
        # Prepare table data
        table_data = []
        headers = ["Metric", "Variant", "Effect Size", "P-Value", "Control Mean", "Treatment Mean", "Lift %", "Success"]
        
        for metric in self.config.metrics:
            if f"{metric}_stats" in results:
                variant_stats = results[f"{metric}_stats"]
                variant_success = results[f"{metric}_success"]
                
                # Add row for each variant
                for variant in self.config.variants:
                    if variant == 'control':
                        continue
                    
                    stats = variant_stats[variant]
                    table_data.append([
                        metric,
                        variant,
                        f"{stats['effect_size']:.4f}",
                        f"{stats['p_value']:.4f}",
                        f"{stats['control_mean']:.2f}",
                        f"{stats['treatment_mean']:.2f}",
                        f"{stats['lift_percentage']:.1f}%",
                        "✓" if variant_success[variant] else "✗"
                    ])
                
                # Add separator between metrics
                if metric != self.config.metrics[-1]:
                    table_data.append(["-" * len(col) for col in headers])
        
        # Add overall result
        table_data.append(["-" * len(col) for col in headers])
        table_data.append([
            "Overall",
            "All Variants",
            "-",
            "-",
            "-",
            "-",
            "-",
            "✓" if results["overall_success"] else "✗"
        ])
        
        # Display the table
        table = tabulate(table_data, headers=headers, tablefmt="grid")
        logger.info("\nExperiment Results Summary:\n" + table)

    def _determine_overall_success(self, results: Dict[str, Any]) -> bool:
        """Determine if the experiment was successful overall."""
        # Check if any variant shows significant improvement in any metric
        has_significant_improvement = False
        has_significant_degradation = False
        
        for metric in self.config.metrics:
            for variant, stats in results[f'{metric}_stats'].items():
                if variant == 'control':
                    continue
                    
                # Check for significant improvement
                if (stats['p_value'] < 0.05 and 
                    stats['effect_size'] > 0.2 and 
                    stats['lift_percentage'] > 0):
                    has_significant_improvement = True
                
                # Check for significant degradation
                if (stats['p_value'] < 0.05 and 
                    (stats['effect_size'] < -0.2 or 
                     stats['lift_percentage'] < -5)):  # 5% degradation threshold
                    has_significant_degradation = True
        
        # Experiment is successful if:
        # 1. At least one metric in one variant shows significant improvement
        # 2. No metric in any variant shows significant degradation
        # 3. No concerning negative trends even if not significant
        return (has_significant_improvement and 
                not has_significant_degradation)

def run_enhanced_loyalty_experiment():
    """Run enhanced experiment with all features."""
    logger.info("Starting enhanced loyalty experiment")
    
    # Configure experiment
    config = AdvancedExperimentConfig(
        name="premium_rewards_test",
        type=ExperimentType.BANDIT,
        variants=["control", "premium_rewards", "premium_plus"],
        metrics=["purchase_amount", "visit_frequency", "satisfaction_score"],
        duration_days=30,
        confidence_level=0.95,
        persistence_path="loyalty_experiments.db",
        minimum_sample_size=100,
        visualization_path="experiment_results",
        bandit_strategy=BanditStrategy.THOMPSON,
        sequential_test=SequentialTestType.OBRIEN_FLEMING,
        cross_validation_folds=5
    )
    logger.info(f"Configured experiment: {config.name}")
    
    # Generate sample data
    logger.info("Generating sample data")
    customers_df = pl.DataFrame({
        'customer_id': [f'cust_{i}' for i in range(1000)],
        'tier': np.random.choice(['gold', 'silver'], size=1000).tolist(),
        'active_status': np.random.choice([True, False], size=1000).tolist()
    })
    
    # Initialize experiment
    experiment = EnhancedLoyaltyExperiment(config)
    logger.info("Initialized enhanced loyalty experiment")
    
    # Run experiment
    try:
        experiment.start_experiment(customers_df)
        logger.info("Successfully started experiment")
        
        # Simulate metrics
        logger.info("Simulating experiment metrics")
        metrics_data = pl.DataFrame({
            'customer_id': [f'cust_{i}' for i in range(1000)],
            'variant': np.random.choice(config.variants, size=1000).tolist(),
            'purchase_amount': np.random.normal(100, 20, 1000).tolist(),
            'visit_frequency': np.random.poisson(5, 1000).tolist(),
            'satisfaction_score': np.random.normal(8, 1, 1000).tolist()
        })
        
        # Analyze results
        logger.info("Analyzing experiment results")
        results = experiment.analyze_results(metrics_data)
        logger.info("Analysis complete")
        
        return results
        
    except Exception as e:
        logger.error(f"Error running experiment: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting main execution")
        results = run_enhanced_loyalty_experiment()
        logger.info("Enhanced Experiment Results:")
        for metric, result in results.items():
            logger.info(f"{metric}: {result}")
    except Exception as e:
        logger.error("Failed to run experiment", exc_info=True)
        raise