"""
READ-ONLY: Basic agent interface.
"""
from typing import Any

class BaseAgent(object):
    def __init__(self, **kwargs: Any) -> None:
        super(BaseAgent, self).__init__(**kwargs)

    def train(self) -> dict:
        """Return a dictionary of logging information."""
        raise NotImplementedError

    def add_to_replay_buffer(self, paths: list) -> None:
        raise NotImplementedError

    def sample(self, batch_size: int) -> tuple:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError
