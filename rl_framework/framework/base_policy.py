import abc
from typing import Any
import torch

class BasePolicyNetwork(torch.nn.Module, abc.ABC):
    """
    Abstract base class for a reinforcement learning policy network.
    It inherits from torch.nn.Module and abc.ABC.
    """

    @abc.abstractmethod
    def forward(self, observation: Any) -> Any:
        """
        Defines the forward pass of the policy network.

        Args:
            observation (Any): The input observation from the environment.
                This can be a tensor or a more complex structure (e.g., dict).

        Returns:
            Any: The action or action distribution. This could be a direct action
                 (e.g., a tensor) or a distribution object from a library like
                 torchrl.modules.distributions.DistributionModule.
        """
        raise NotImplementedError
