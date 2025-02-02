from typing import Dict, Any
from datetime import datetime
import sqlite3
import json
import polars as pl
from pathlib import Path

from ..core.config import BaseExperimentConfig

class DataPersistence:
    """Handle experiment data persistence using SQLite."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """Create necessary tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    config TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    status TEXT NOT NULL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS assignments (
                    experiment_id INTEGER NOT NULL,
                    customer_id TEXT NOT NULL,
                    variant TEXT NOT NULL,
                    assignment_time TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id),
                    PRIMARY KEY (experiment_id, customer_id)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    experiment_id INTEGER NOT NULL,
                    customer_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            ''')
    
    def save_experiment(self, config: BaseExperimentConfig, start_time: datetime) -> int:
        """Save experiment configuration and return experiment ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                '''
                INSERT INTO experiments (name, config, start_time, status)
                VALUES (?, ?, ?, ?)
                ''',
                (
                    config.name,
                    json.dumps(self._config_to_dict(config)),
                    start_time.isoformat(),
                    'RUNNING'
                )
            )
            return cursor.lastrowid
    
    def save_assignments(self, experiment_id: int, assignments: Dict[str, str]) -> None:
        """Save variant assignments."""
        assignment_time = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                '''
                INSERT INTO assignments (experiment_id, customer_id, variant, assignment_time)
                VALUES (?, ?, ?, ?)
                ''',
                [
                    (experiment_id, customer_id, variant, assignment_time)
                    for customer_id, variant in assignments.items()
                ]
            )
    
    def save_metrics(self, experiment_id: int, metrics_data: pl.DataFrame) -> None:
        """Save metrics data."""
        timestamp = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            for metric in metrics_data.columns:
                if metric not in ['customer_id', 'variant']:
                    data = [
                        (experiment_id, row['customer_id'], metric, row[metric], timestamp)
                        for row in metrics_data.select(['customer_id', pl.col(metric)]).iter_rows(named=True)
                    ]
                    conn.executemany(
                        '''
                        INSERT INTO metrics (experiment_id, customer_id, metric_name, value, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                        ''',
                        data
                    )
    
    def get_experiment_results(self, experiment_id: int) -> Dict[str, Any]:
        """Retrieve experiment results."""
        with sqlite3.connect(self.db_path) as conn:
            # Get experiment config
            config_row = conn.execute(
                'SELECT config FROM experiments WHERE id = ?',
                (experiment_id,)
            ).fetchone()
            
            if not config_row:
                raise ValueError(f"No experiment found with id {experiment_id}")
            
            config = json.loads(config_row[0])
            
            # Get assignments
            assignments_df = pl.read_database(
                query='SELECT customer_id, variant FROM assignments WHERE experiment_id = ?',
                connection=conn,
                params=(experiment_id,)
            )
            
            # Get metrics
            metrics_df = pl.read_database(
                query='SELECT customer_id, metric_name, value FROM metrics WHERE experiment_id = ?',
                connection=conn,
                params=(experiment_id,)
            )
            
            return {
                'config': config,
                'assignments': assignments_df,
                'metrics': metrics_df
            }
    
    @staticmethod
    def _config_to_dict(config: BaseExperimentConfig) -> Dict[str, Any]:
        """Convert experiment config to dictionary for storage."""
        return {
            'name': config.name,
            'type': config.type.value,
            'variants': config.variants,
            'metrics': config.metrics,
            'duration_days': config.duration_days,
            'confidence_level': config.confidence_level,
            'minimum_sample_size': config.minimum_sample_size,
            'persistence_path': str(config.persistence_path) if config.persistence_path else None
        }
