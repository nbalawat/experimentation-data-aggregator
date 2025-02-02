from enum import Enum, auto

class ExperimentType(Enum):
    """Types of experiments supported by the framework."""
    AB_TEST = auto()
    MULTI_VARIATE = auto()
    BANDIT = auto()
    FACTORIAL = auto()

class BanditStrategy(Enum):
    """Available strategies for bandit experiments."""
    THOMPSON = auto()
    UCB = auto()
    EPSILON_GREEDY = auto()

class SequentialTestType(Enum):
    """Types of sequential testing procedures."""
    OBRIEN_FLEMING = auto()
    POCOCK = auto()
    HAYBITTLE_PETO = auto()
