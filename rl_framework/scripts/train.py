import argparse
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from rl_framework.framework.ppo_agent import PPOAgent

# --- Helper function to parse arguments ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train a PPO agent using PyTorch Lightning.")
    parser.add_argument(
        "--env_name", 
        type=str, 
        choices=["cartpole", "custom_env"], 
        required=True,
        help="Name of the environment to train on."
    )
    parser.add_argument(
        "--max_epochs", 
        type=int, 
        default=100, 
        help="Number of training epochs."
    )
    parser.add_argument(
        "--num_gpus", 
        type=int, 
        default=0, 
        help="Number of GPUs to use (0 for CPU)."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="Random seed for reproducibility. If None, no seed is set explicitly beyond Lightning's."
    )
    
    # PPO Agent specific hyperparameters
    parser.add_argument("--lr_policy", type=float, default=3e-4, help="Learning rate for the policy network.")
    parser.add_argument("--lr_value", type=float, default=1e-3, help="Learning rate for the value network.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda parameter.")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO clip epsilon.")
    parser.add_argument("--ppo_epochs", type=int, default=10, help="Number of PPO epochs per data collection.")
    parser.add_argument("--num_rollout_steps", type=int, default=2048, help="Steps per environment before PPO update.")
    parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size for PPO updates.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient in PPO loss.")
    parser.add_argument("--value_loss_coef", type=float, default=0.5, help="Value loss coefficient in PPO loss.")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm for clipping.")
    
    return parser.parse_args()

# --- Main training function ---
def main():
    args = parse_args()

    if args.seed is not None:
        L.seed_everything(args.seed, workers=True) # workers=True ensures dataloader reproducibility

    # Determine device for PPOAgent initialization (Trainer will manage actual device placement)
    ppo_device = 'cuda' if args.num_gpus > 0 and torch.cuda.is_available() else 'cpu'

    # --- Environment and Network Setup ---
    if args.env_name == 'cartpole':
        from rl_framework.examples.cartpole.cartpole_env import CartPoleEnvWrapper
        from rl_framework.examples.cartpole.cartpole_networks import CartPolePolicy, CartPoleValue
        
        # Pass device to env_factory if the env wrapper supports it
        env_factory = lambda: CartPoleEnvWrapper(device=ppo_device) 
        temp_env = env_factory()
        obs_spec = temp_env.observation_spec()
        action_spec = temp_env.action_spec()
        temp_env.close()

        policy_network = CartPolePolicy(observation_spec=obs_spec, action_spec=action_spec)
        value_network = CartPoleValue(observation_spec=obs_spec)

    elif args.env_name == 'custom_env':
        from rl_framework.examples.custom_env.custom_environment import CustomComplexEnv
        from rl_framework.examples.custom_env.custom_networks import CustomPolicyNetwork, CustomValueNetwork
        
        env_factory = lambda: CustomComplexEnv(device=ppo_device)
        temp_env = env_factory()
        obs_spec = temp_env.observation_spec()
        action_spec = temp_env.action_spec()
        temp_env.close()

        policy_network = CustomPolicyNetwork(observation_spec=obs_spec, action_spec=action_spec)
        value_network = CustomValueNetwork(observation_spec=obs_spec)
    else:
        raise ValueError(f"Unknown environment: {args.env_name}")

    # --- Instantiate PPOAgent ---
    ppo_agent = PPOAgent(
        env_factory=env_factory,
        policy_network=policy_network,
        value_network=value_network,
        lr_policy=args.lr_policy,
        lr_value=args.lr_value,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        ppo_epochs=args.ppo_epochs,
        num_rollout_steps=args.num_rollout_steps,
        batch_size=args.batch_size,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        max_grad_norm=args.max_grad_norm,
        device=ppo_device # Initial device hint for the agent
    )

    # --- Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        monitor="info/mean_reward_rollout", # Metric to monitor
        mode="max",                        # Maximize the metric
        save_top_k=1,
        filename=f"{args.env_name}-ppo-{{epoch:02d}}-{{info/mean_reward_rollout:.2f}}",
        dirpath=f"checkpoints/{args.env_name}_ppo/"
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # --- Instantiate Trainer ---
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if args.num_gpus > 0 and torch.cuda.is_available() else 'cpu',
        devices=args.num_gpus if args.num_gpus > 0 and torch.cuda.is_available() else 1, # 1 for cpu
        callbacks=[checkpoint_callback, lr_monitor],
        # deterministic=True if args.seed is not None else False, # For stricter reproducibility
        logger=True, # Enables default TensorBoardLogger or other configured loggers
        log_every_n_steps=10 # How often to log metrics
    )

    # --- Start Training ---
    print(f"Starting training for {args.env_name} with PPOAgent on {ppo_device.upper()}...")
    trainer.fit(ppo_agent)

    print("Training complete.")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")

if __name__ == '__main__':
    main()
