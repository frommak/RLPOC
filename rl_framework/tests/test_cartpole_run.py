import pytest
import torch
import lightning as L
from rl_framework.framework.ppo_agent import PPOAgent
from rl_framework.examples.cartpole.cartpole_env import CartPoleEnvWrapper
from rl_framework.examples.cartpole.cartpole_networks import CartPolePolicy, CartPoleValue

def test_cartpole_ppo_single_epoch_run():
    """
    Tests a single epoch run of PPOAgent with CartPole.
    This verifies that the training_step and data collection can execute.
    """
    device = "cpu"
    
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
    
    # 4. Instantiate PPOAgent with minimal parameters for a quick test
    agent = PPOAgent(
        env_factory=env_factory,
        policy_network=policy_network,
        value_network=value_network,
        device=device,
        num_rollout_steps=16, # Minimal steps to collect some data
        batch_size=4,         # Small batch size
        ppo_epochs=1,         # Minimal PPO epochs
        lr_policy=1e-5,       # Small LR to avoid large updates in a short test
        lr_value=1e-5
    )
    
    # 5. Instantiate Trainer
    # Use fast_dev_run for a very quick test of one batch.
    # For a slightly more comprehensive test of one epoch, set max_epochs=1.
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False, # Disable logging for tests
        enable_checkpointing=False, # Disable checkpointing for tests
        enable_progress_bar=False, # Disable progress bar for cleaner test output
        enable_model_summary=False # Disable model summary
    )
    
    # 6. Call trainer.fit()
    try:
        trainer.fit(agent)
        print("PPOAgent CartPole single epoch run test passed.")
    except Exception as e:
        pytest.fail(f"trainer.fit() with PPOAgent and CartPole failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
