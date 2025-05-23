import pytest
import torch
from rl_framework.framework.ppo_agent import PPOAgent
from rl_framework.examples.cartpole.cartpole_env import CartPoleEnvWrapper
from rl_framework.examples.cartpole.cartpole_networks import CartPolePolicy, CartPoleValue

def test_ppo_agent_cartpole_initialization():
    """
    Tests the basic initialization of PPOAgent with CartPole components.
    """
    device = "cpu" # Tests should run on CPU for simplicity and portability
    
    # 1. Create CartPoleEnvWrapper factory
    env_factory = lambda: CartPoleEnvWrapper(device=device)
    
    # 2. Create a temporary CartPoleEnvWrapper instance to get specs
    temp_env = env_factory()
    obs_spec = temp_env.observation_spec()
    action_spec = temp_env.action_spec()
    temp_env.close()
    
    # 3. Instantiate CartPolePolicy and CartPoleValue
    policy_network = CartPolePolicy(observation_spec=obs_spec, action_spec=action_spec)
    value_network = CartPoleValue(observation_spec=obs_spec)
    
    # 4. Instantiate PPOAgent
    agent = PPOAgent(
        env_factory=env_factory,
        policy_network=policy_network,
        value_network=value_network,
        device=device,
        # Using default hyperparameters for other PPO params for this init test
        num_rollout_steps=128, # Small value for testing
        batch_size=32,
        ppo_epochs=4
    )
    
    # 5. Assertions
    assert agent is not None, "PPOAgent failed to initialize."
    assert agent.policy_network == policy_network, "Policy network in agent is not the one passed."
    assert agent.value_network == value_network, "Value network in agent is not the one passed."
    assert agent.hparams.device == device, f"Agent device is {agent.hparams.device}, expected {device}"

    print("PPOAgent CartPole initialization test passed.")

if __name__ == "__main__":
    pytest.main([__file__])
