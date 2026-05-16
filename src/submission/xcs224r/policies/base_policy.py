"""
READ-ONLY: Basic policy interface
"""
import abc
import numpy as np
from typing import Any

class BasePolicy(object, metaclass=abc.ABCMeta):
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(self, obs: np.ndarray, acs: np.ndarray, **kwargs: Any) -> dict:
        """Return a dictionary of logging information."""
        raise NotImplementedError

    def save(self, filepath: str) -> None:
        raise NotImplementedError
