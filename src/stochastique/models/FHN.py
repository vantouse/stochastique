import numpy as np
from typing import List, Dict, Any
from .base import DynamicalSystemModel


def model_FHN(
    time: float,
    state: np.ndarray,
    delta: float = 0.1,
    a: float = 1,
) -> np.ndarray:
    """
    FitzHugh-Nagumo neural model.
    Accepts `time` argument for interface consistency, but does not use it.
    
    Parameter restrictions:
    - delta << 1 (delta = 0.1)
    - a > 0
    """
    x, y = state
    return np.array([
        (x - (x ** 3) / 3 - y) / delta,
        x + a
    ])


def equilibria_FHN(
    delta: float = 0.1,
    a: float = 1,
) -> List[np.ndarray]:
    """
    Return all the real equilibrium values of the FitzHugh-Nagumo model at given parameter values.
    """
    return [
        np.array([
            -a,
            a**3 / 3 - a
        ])
    ]


class FitzHughNagumo(DynamicalSystemModel):
    """Object-oriented wrapper for the FitzHugh-Nagumo model."""

    @property
    def default_params(self) -> Dict[str, float]:
        return {"delta": 0.1, "a": 1.0}

    def __call__(self, time: float, state: np.ndarray, **kwargs: Any) -> np.ndarray:
        params = {**self.default_params, **kwargs}
        return model_FHN(time, state, **params)

    def equilibria(self, **kwargs: Any) -> List[np.ndarray]:
        params = {**self.default_params, **kwargs}
        return equilibria_FHN(**params)
