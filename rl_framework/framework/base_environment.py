import abc
from typing import Any
import torch # Added import for torch.Size

class BaseEnvironment(abc.ABC):
    """
    Abstract base class for a reinforcement learning environment.
    """
    _has_dynamic_specs = False # Default for most custom envs
    batch_size = torch.Size()   # For single envs, batch_size is empty

    @abc.abstractmethod
    def reset(self, seed: int = None) -> Any:
        """
        Resets the environment to an initial state.

        Args:
            seed (int, optional): The seed to use for the environment's random
                number generator. Defaults to None.

        Returns:
            Any: The initial observation from the environment.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """
        Takes an action in the environment and advances it by one step.

        Args:
            action (Any): The action to take in the environment.

        Returns:
            tuple[Any, float, bool, bool, dict]: A tuple containing:
                - observation (Any): The observation from the environment after the step.
                - reward (float): The reward received from the environment after the step.
                - terminated (bool): Whether the episode has terminated.
                - truncated (bool): Whether the episode has been truncated (e.g., due to time limit).
                - info (dict): Additional information from the environment.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """
        Closes the environment and cleans up any resources.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def observation_spec(self) -> Any:
        """
        Returns the specification of the observation space.

        This defines the structure, shape, and type of the observations
        that the environment can produce.

        Returns:
            Any: The observation space specification.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def action_spec(self) -> Any:
        """
        Returns the specification of the action space.

        This defines the structure, shape, and type of the actions
        that can be taken in the environment.

        Returns:
            Any: The action space specification.
        """
        raise NotImplementedError
