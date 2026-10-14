import numpy as np
from typing import List, Dict, Any
from .base import DynamicalSystemModel


def model_SN(
    time: float,
    state: np.ndarray,
    a: float = 1,
    b: float = 2,
) -> np.ndarray:
    """
    Saltzman-Nicolis climate model.
    Accepts `time` argument for interface consistency, but does not use it.
    
    Parameter restrictions:
    - a > 0
    - b = 2
    """
    x, y = state
    return np.array([
        y - x,
        -a * x + b * y - (x ** 2) * y
    ])


def equilibria_SN(
    a: float = 1,
    b: float = 2,
) -> List[np.ndarray]:
    """
    Return all the real equilibrium values of the Saltzman-Nicolis model at given parameter values.
    """
    return [
        np.array([
            np.sqrt(b - a),
            np.sqrt(b - a)
        ]),
        np.array([
            0,
            0
        ]),
        np.array([
            -np.sqrt(b - a),
            -np.sqrt(b - a)
        ]),
    ]


class SaltzmanNicolis(DynamicalSystemModel):
    """Object-oriented wrapper for the Saltzman-Nicolis model."""

    @property
    def default_params(self) -> Dict[str, float]:
        return {"a": 1.0, "b": 2.0}

    def __call__(self, time: float, state: np.ndarray, **kwargs: Any) -> np.ndarray:
        params = {**self.default_params, **kwargs}
        return model_SN(time, state, **params)

    def equilibria(self, **kwargs: Any) -> List[np.ndarray]:
        params = {**self.default_params, **kwargs}
        return equilibria_SN(**params)
