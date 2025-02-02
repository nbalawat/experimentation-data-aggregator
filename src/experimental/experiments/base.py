from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import polars as pl
from datetime import datetime
from pathlib import Path

from ..core.config import BaseExperimentConfig
from ..persistence.database import DataPersistence
from ..utils.logging import ExperimentLogger

class ExperimentGroup(ABC):
    """Abstract base class for different types of experiment groups."""
    
    def __init__(self, config: BaseExperimentConfig):
        self.config = config
        self.logger = ExperimentLogger(
            f"{config.name}.{self.__class__.__name__}",
            log_file=Path("logs") / f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
    @abstractmethod
    def assign_variants(self, customer_ids: List[str]) -> Dict[str, str]:
        """Assign variants to customers."""
        pass
    
    @abstractmethod
    def calculate_results(self, metrics_data: pl.DataFrame) -> Dict[str, Any]:
        """Calculate experiment results."""
        pass

class BaseExperiment(ABC):
    """Base class for all experiment types."""
    
    def __init__(
        self,
        config: BaseExperimentConfig,
        persistence: Optional[DataPersistence] = None
    ):
        self.config = config
        self.persistence = persistence
        self.start_time: Optional[datetime] = None
        self.assignments: Dict[str, str] = {}
        self.experiment_id: Optional[int] = None
        self.logger = ExperimentLogger(
            config.name,
            log_file=Path("logs") / f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
    @abstractmethod
    def _create_experiment_group(self) -> ExperimentGroup:
        """Factory method to create appropriate experiment group."""
        pass
    
    def start_experiment(self, customers_df: pl.DataFrame) -> None:
        """Initialize and start the experiment."""
        self.logger.experiment_start({
            'name': self.config.name,
            'type': self.config.type.value,
            'variants': self.config.variants,
            'metrics': self.config.metrics,
            'duration_days': self.config.duration_days,
            'sample_size': customers_df.height
        })
        
        if customers_df.height < self.config.minimum_sample_size:
            self.logger.error(
                f"Insufficient sample size: {customers_df.height} < "
                f"{self.config.minimum_sample_size}"
            )
            raise ValueError(
                f"Insufficient sample size: {customers_df.height} < "
                f"{self.config.minimum_sample_size}"
            )
        
        self.start_time = datetime.now()
        experiment_group = self._create_experiment_group()
        
        self.logger.info("Assigning variants to customers...")
        self.assignments = experiment_group.assign_variants(
            customers_df.select('customer_id').to_series().to_list()
        )
        
        variant_counts = {
            variant: list(self.assignments.values()).count(variant)
            for variant in self.config.variants
        }
        self.logger.info("Variant assignment complete:")
        for variant, count in variant_counts.items():
            self.logger.info(f"  {variant}: {count} customers")
        
        if self.persistence:
            self.logger.info("Saving experiment configuration and assignments...")
            self.experiment_id = self.persistence.save_experiment(
                self.config,
                self.start_time
            )
            self.persistence.save_assignments(
                self.experiment_id,
                self.assignments
            )
            self.logger.info(f"Experiment saved with ID: {self.experiment_id}")
    
    def record_metrics(self, metrics_data: pl.DataFrame) -> None:
        """Record experiment metrics."""
        if not self.start_time:
            self.logger.error("Cannot record metrics: Experiment hasn't started yet")
            raise ValueError("Experiment hasn't started yet")
        
        self.logger.info(f"Recording metrics for {metrics_data.height} customers")
        metrics_summary = {
            metric: {
                'mean': metrics_data.select(pl.col(metric).mean()).item(),
                'std': metrics_data.select(pl.col(metric).std()).item(),
                'min': metrics_data.select(pl.col(metric).min()).item(),
                'max': metrics_data.select(pl.col(metric).max()).item()
            }
            for metric in self.config.metrics
            if metric in metrics_data.columns
        }
        self.logger.experiment_metrics(metrics_summary)
            
        if self.persistence and self.experiment_id:
            self.logger.info("Saving metrics to persistence layer...")
            self.persistence.save_metrics(
                self.experiment_id,
                metrics_data
            )
            self.logger.info("Metrics saved successfully")
    
    @abstractmethod
    def analyze_results(self, metrics_data: pl.DataFrame) -> Dict[str, Any]:
        """Analyze experiment results."""
        pass
