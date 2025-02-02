"""Logging configuration and utilities for the experimental framework."""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

class ExperimentLogger:
    """Configure and manage logging for experiments."""
    
    def __init__(
        self,
        name: str,
        log_file: Optional[Path] = None,
        level: int = logging.INFO,
        format_string: Optional[str] = None
    ):
        """Initialize logger with custom configuration.
        
        Args:
            name: Name of the logger, typically experiment name
            log_file: Optional path to log file
            level: Logging level (default: INFO)
            format_string: Optional custom format string for log messages
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        if not format_string:
            format_string = (
                '%(asctime)s [%(levelname)s] %(name)s - '
                '%(filename)s:%(lineno)d - %(message)s'
            )
        
        formatter = logging.Formatter(format_string)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if log_file is provided
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def experiment_start(self, config_details: dict) -> None:
        """Log experiment start with configuration details."""
        self.logger.info("=" * 80)
        self.logger.info("Starting Experiment: %s", config_details.get('name', 'Unknown'))
        self.logger.info("Time: %s", datetime.now().isoformat())
        self.logger.info("Configuration:")
        for key, value in config_details.items():
            self.logger.info("  %s: %s", key, value)
        self.logger.info("=" * 80)
    
    def experiment_progress(
        self,
        phase: str,
        current: int,
        total: int,
        additional_info: Optional[dict] = None
    ) -> None:
        """Log experiment progress with completion percentage."""
        percentage = (current / total) * 100 if total > 0 else 0
        progress_msg = f"{phase}: {current}/{total} ({percentage:.1f}%)"
        if additional_info:
            progress_msg += " - " + ", ".join(
                f"{k}: {v}" for k, v in additional_info.items()
            )
        self.logger.info(progress_msg)
    
    def experiment_metrics(self, metrics: dict) -> None:
        """Log experiment metrics."""
        self.logger.info("-" * 40)
        self.logger.info("Experiment Metrics:")
        for metric_name, value in metrics.items():
            self.logger.info("  %s: %s", metric_name, value)
        self.logger.info("-" * 40)
    
    def experiment_complete(self, summary: dict) -> None:
        """Log experiment completion with summary."""
        self.logger.info("=" * 80)
        self.logger.info("Experiment Complete")
        self.logger.info("Time: %s", datetime.now().isoformat())
        self.logger.info("Summary:")
        for key, value in summary.items():
            self.logger.info("  %s: %s", key, value)
        self.logger.info("=" * 80)
    
    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs) -> None:
        """Log error message."""
        self.logger.error(msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs) -> None:
        """Log info message."""
        self.logger.info(msg, *args, **kwargs)
