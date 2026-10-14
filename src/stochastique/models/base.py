# stochastique/models/base.py
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any


class DynamicalSystemModel(ABC):
    """Abstract base class for 2D dynamical system models."""

    @property
    @abstractmethod
    def default_params(self) -> Dict[str, float]:
        """Return default model parameters."""
        pass

    @abstractmethod
    def __call__(self, time: float, state: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute system right-hand side derivatives (dx/dt, dy/dt)."""
        pass

    @abstractmethod
    def equilibria(self, **kwargs: Any) -> List[np.ndarray]:
        """Compute analytical real equilibria of the model."""
        pass
