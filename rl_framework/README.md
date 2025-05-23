# RL PPO Framework

## Overview

This project provides a generic Proximal Policy Optimization (PPO) framework designed for reinforcement learning tasks. It aims to be reusable and extensible, supporting environments with complex state and action spaces. The framework leverages `torchrl` for RL-specific components (like data collection and distributions) and `pytorch-lightning` for structuring the training loop and hardware abstraction.

## Project Structure

The repository is organized as follows:

*   `framework/`: Contains the core PPO agent implementation (`ppo_agent.py`), abstract base classes for environments (`base_environment.py`), policy networks (`base_policy.py`), and value networks (`base_value.py`).
*   `examples/`: Includes sample environment implementations and corresponding policy/value networks.
    *   `cartpole/`: A classic CartPole environment example.
    *   `custom_env/`: An example of a custom environment with a complex (dictionary) observation space and a composite action space.
*   `scripts/`: Provides command-line scripts for training (`train.py`) and running inference (`infer.py`) with the PPO agent.
*   `tests/`: Contains unit tests for the framework components and examples.
*   `requirements.txt`: Lists the Python dependencies required for the project.
*   `checkpoints/`: Default directory where trained model checkpoints are saved.

## Setup Instructions

1.  **Create a Virtual Environment (Recommended)**:
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python3 -m venv rl_env
    source rl_env/bin/activate 
    ```
    (On Windows, use `rl_env\Scripts\activate`)

2.  **Install Dependencies**:
    Install all required packages using the `requirements.txt` file:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

## Running Examples

### Training

The main script for training is `scripts/train.py`.

*   **General Command**:
    ```bash
    python scripts/train.py --env_name <environment_name> [OPTIONS]
    ```
    Replace `<environment_name>` with either `cartpole` or `custom_env`.

*   **Example for CartPole**:
    Train the PPO agent on the CartPole environment for 50 epochs with a specific seed:
    ```bash
    python scripts/train.py --env_name cartpole --max_epochs 50 --seed 42
    ```

*   **Example for Custom Environment**:
    Train the PPO agent on the custom environment for 100 epochs, with 512 rollout steps per data collection, and a specific seed:
    ```bash
    python scripts/train.py --env_name custom_env --max_epochs 100 --num_rollout_steps 512 --seed 42
    ```

*   **Checkpoints**:
    Trained model checkpoints will be saved in the `checkpoints/<environment_name>_ppo/` directory by default. The best model based on `info/mean_reward_rollout` will be saved.

### Inference

The main script for running inference with a trained model is `scripts/infer.py`.

*   **General Command**:
    ```bash
    python scripts/infer.py --checkpoint_path <path_to_checkpoint.ckpt> --env_name <environment_name> [OPTIONS]
    ```

*   **Example**:
    Run inference with a trained CartPole model, render the environment, and run for 10 episodes:
    ```bash
    # Replace XX with the actual epoch number and ... with the rest of the checkpoint filename
    python scripts/infer.py --checkpoint_path checkpoints/cartpole_ppo/cartpole-ppo-epoch=XX-....ckpt --env_name cartpole --num_episodes 10 --render
    ```

## Extending the Framework

### Custom Environments

To implement a new environment:

1.  Create a new Python class that inherits from `rl_framework.framework.base_environment.BaseEnvironment`.
2.  Implement the following abstract methods:
    *   `reset(self, seed=None)`: Resets the environment and returns the initial observation.
    *   `step(self, action)`: Applies an action and returns `(observation, reward, terminated, truncated, info)`.
    *   `close(self)`: Cleans up environment resources.
    *   `observation_spec(self)` (as a `@property`): Returns the observation spec (e.g., `UnboundedContinuousTensorSpec`, `DiscreteTensorSpec`, `CompositeSpec`).
    *   `action_spec(self)` (as a `@property`): Returns the action spec.
3.  Ensure your environment defines `_has_dynamic_specs = False` (or `True` if applicable) and `batch_size = torch.Size()` (for single envs) as class attributes or properties.
4.  For complex state or action spaces, use `torchrl.data.CompositeSpec`. Ensure your `reset` and `step` methods correctly produce and consume `TensorDict` objects for observations and actions if using `CompositeSpec`. Refer to `examples/custom_env/custom_environment.py` for an example.

### Custom Networks

To implement new policy or value networks:

1.  **Policy Network**:
    *   Create a class inheriting from `rl_framework.framework.base_policy.BasePolicyNetwork`.
    *   Implement the `__init__` method, accepting `observation_spec`, `action_spec`, and optionally `device`.
    *   Implement the `forward(self, observation)` method.
        *   This method should process the input `observation` (which can be a tensor or a `TensorDict` for complex observation spaces).
        *   It must return an appropriate TorchRL distribution object (e.g., `OneHotCategorical` for discrete actions, or the output of `NormalParamWrapper` for continuous actions).

2.  **Value Network**:
    *   Create a class inheriting from `rl_framework.framework.base_value.BaseValueNetwork`.
    *   Implement the `__init__` method, accepting `observation_spec` and optionally `device`.
    *   Implement the `forward(self, observation)` method.
        *   This method should process the input `observation`.
        *   It must return a tensor representing the state value, typically of shape `[batch_size, 1]`.

Refer to `examples/cartpole/cartpole_networks.py` and `examples/custom_env/custom_networks.py` for examples.

## Running Tests

Unit tests are located in the `rl_framework/tests/` directory. To discover and run them:

1.  Ensure `pytest` is installed (it's included in `requirements.txt`).
2.  Navigate to the root of the repository (the directory containing `rl_framework`).
3.  Run `pytest`:
    ```bash
    # Ensure PYTHONPATH includes the root directory if running from elsewhere
    # export PYTHONPATH=/path/to/your/project_root:$PYTHONPATH 
    pytest rl_framework/tests
    ```
    Alternatively, using Python's built-in `unittest` discovery (though tests are written in `pytest` style):
    ```bash
    python -m unittest discover -s rl_framework/tests
    ```

This framework provides a solid foundation for PPO-based reinforcement learning research and development.
