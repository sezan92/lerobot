from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import torch
from torch import Tensor


class RLAlgorithm(ABC):
    """Abstract base class for RL algorithms in LeRobot.

    Each subclass wraps a policy (neural network model) and owns:
    - its optimizers
    - its update logic (including UTD ratio)
    - how to push/pull weights to/from actors

    The learner and actor only interact through this interface,
    making them policy-agnostic.
    """

    @abstractmethod
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select an action for environment interaction (actor side).

        Args:
            batch: Dictionary of observation tensors.

        Returns:
            Action tensor.
        """
        ...

    @abstractmethod
    def update(self, batch_iterator: Iterator[dict]) -> dict[str, float]:
        """Perform one full training update (learner side).

        The algorithm controls how many batches it pulls from the iterator.
        This is where UTD ratio and multi-phase updates live.

        Args:
            batch_iterator: Iterator that yields batches from the replay buffer.

        Returns:
            Dictionary of training metrics (losses, grad norms, etc.) for logging.
        """
        ...

    @abstractmethod
    def get_weights(self) -> dict[str, dict]:
        """Return the weights to push to actors after a training update.

        Returns:
            Dictionary mapping component names to state dicts.
            Example: {"policy": actor_state_dict}
        """
        ...

    @abstractmethod
    def load_weights(self, weights: dict[str, dict]) -> None:
        """Load weights received from the learner (actor side).

        Args:
            weights: Same format as returned by get_weights().
        """
        ...

    @abstractmethod
    def to(self, device: torch.device | str) -> "RLAlgorithm":
        """Move all model components to the specified device."""
        ...

    def train(self) -> None:
        """Set all model components to training mode."""
        pass

    def eval(self) -> None:
        """Set all model components to eval mode."""
        pass