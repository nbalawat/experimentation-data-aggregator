from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

from .enums import ExperimentType, BanditStrategy, SequentialTestType

@dataclass
class BaseExperimentConfig:
    """Base configuration for all experiments."""
    name: str
    type: ExperimentType
    variants: List[str]
    metrics: List[str]
    duration_days: int
    confidence_level: float = 0.95
    minimum_sample_size: int = 100
    persistence_path: Optional[Path] = None

@dataclass
class BanditConfig:
    """Configuration specific to bandit experiments."""
    strategy: BanditStrategy
    epsilon: float = 0.1  # For epsilon-greedy
    ucb_alpha: float = 1.0  # For UCB
    exploration_rounds: int = 1000

@dataclass
class VisualizationConfig:
    """Configuration for experiment visualization."""
    output_path: Path
    show_distributions: bool = True
    show_convergence: bool = True
    show_boundaries: bool = True

@dataclass
class AdvancedExperimentConfig(BaseExperimentConfig):
    """Enhanced configuration with advanced features."""
    bandit_config: Optional[BanditConfig] = None
    sequential_test: Optional[SequentialTestType] = None
    visualization_config: Optional[VisualizationConfig] = None
    cross_validation_folds: int = 5
