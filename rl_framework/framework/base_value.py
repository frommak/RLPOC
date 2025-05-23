import abc
from typing import Any
import torch

class BaseValueNetwork(torch.nn.Module, abc.ABC):
    """
    Abstract base class for a reinforcement learning value network.
    It inherits from torch.nn.Module and abc.ABC.
    """

    @abc.abstractmethod
    def forward(self, observation: Any) -> torch.Tensor:
        """
        Defines the forward pass of the value network.

        Args:
            observation (Any): The input observation from the environment.
                This can be a tensor or a more complex structure (e.g., dict).

        Returns:
            torch.Tensor: A tensor representing the state value.
                          Expected shape: [batch_size, 1].
        """
        raise NotImplementedError
