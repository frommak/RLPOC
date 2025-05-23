import pytest
import torch
import lightning as L
from rl_framework.framework.ppo_agent import PPOAgent
from rl_framework.examples.custom_env.custom_environment import CustomComplexEnv
from rl_framework.examples.custom_env.custom_networks import CustomPolicyNetwork, CustomValueNetwork

def test_custom_env_ppo_single_epoch_run(): # Changed from single_step to single_epoch_run to match cartpole test
    """
    Tests a single epoch run of PPOAgent with CustomComplexEnv.
    This verifies that the training_step and data collection can execute
    with complex observation and action spaces (TensorDicts).
    """
    device = "cpu"
    
    # 1. Create CustomComplexEnv factory
    env_factory = lambda: CustomComplexEnv(device=device, max_steps=20) # Short max_steps for testing
    
    # 2. Create a temporary CustomComplexEnv instance to get specs
    temp_env = env_factory()
    obs_spec = temp_env.observation_spec()
    action_spec = temp_env.action_spec()
    temp_env.close()
    
    # 3. Instantiate CustomPolicyNetwork and CustomValueNetwork
    # Use smaller networks for faster testing
    policy_network = CustomPolicyNetwork(
        observation_spec=obs_spec, 
        action_spec=action_spec,
        hidden_sizes_cnn=[8], # Single small CNN layer
        hidden_sizes_mlp=[16]  # Single small MLP layer
    )
    value_network = CustomValueNetwork(
        observation_spec=obs_spec,
        hidden_sizes_cnn=[8],
        hidden_sizes_mlp=[16]
    )
    
    # 4. Instantiate PPOAgent with minimal parameters for a quick test
    agent = PPOAgent(
        env_factory=env_factory,
        policy_network=policy_network,
        value_network=value_network,
        device=device,
        num_rollout_steps=10, # Minimal steps to collect some data (must be >= batch_size if not shuffling full buffer)
        batch_size=5,         # Small batch size
        ppo_epochs=1,         # Minimal PPO epochs
        lr_policy=1e-5,
        lr_value=1e-5
    )
    
    # 5. Instantiate Trainer
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False, 
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False 
    )
    
    # 6. Call trainer.fit()
    try:
        trainer.fit(agent)
        print("PPOAgent CustomComplexEnv single epoch run test passed.")
    except Exception as e:
        pytest.fail(f"trainer.fit() with PPOAgent and CustomComplexEnv failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
