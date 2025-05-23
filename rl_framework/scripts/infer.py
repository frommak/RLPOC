import argparse
import torch
import time # For potential sleep during rendering
from rl_framework.framework.ppo_agent import PPOAgent
from torchrl.data import TensorDict # For custom_env observations/actions
from torchrl.data.specs.tensor_specs import UnboundedContinuousTensorSpec, DiscreteTensorSpec, CompositeSpec # For dummy specs

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained PPO agent.")
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        required=True,
        help="Path to the saved model checkpoint (.ckpt file)."
    )
    parser.add_argument(
        "--env_name", 
        type=str, 
        choices=["cartpole", "custom_env"], 
        required=True,
        help="Name of the environment to run inference on."
    )
    parser.add_argument(
        "--num_episodes", 
        type=int, 
        default=10, 
        help="Number of episodes to run inference for."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--render", 
        action='store_true', # Default is False unless specified
        help="Enable rendering if the environment supports it."
    )
    parser.add_argument(
        "--no-render",
        action='store_false',
        dest='render', # Store to 'render'
        help="Disable rendering."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu", 
        choices=["cpu", "cuda"],
        help="Device to run inference on ('cpu' or 'cuda')."
    )
    parser.add_argument(
        "--render_delay",
        type=float,
        default=0.03, # Approx 30 FPS
        help="Delay in seconds between rendered frames."
    )
    # Set default render based on env
    # This is a bit tricky with argparse, so we'll handle it in main()
    return parser.parse_args()

def main():
    args = parse_args()

    # Handle default rendering based on environment
    if args.env_name == "cartpole" and args.render is None: # If render flag not set by user
        args.render = True
    elif args.env_name == "custom_env" and args.render is None:
        args.render = False # CustomEnv has no visual component by default

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # --- Environment and Network Setup ---
    env_factory = None
    policy_network_dummy = None
    value_network_dummy = None

    if args.env_name == 'cartpole':
        from rl_framework.examples.cartpole.cartpole_env import CartPoleEnvWrapper
        from rl_framework.examples.cartpole.cartpole_networks import CartPolePolicy, CartPoleValue
        
        render_mode_arg = 'human' if args.render else None
        env_factory = lambda: CartPoleEnvWrapper(device=str(device), render_mode=render_mode_arg)
        env = env_factory()

        obs_spec_dummy = env.observation_spec().to(device)
        action_spec_dummy = env.action_spec().to(device)
        
        policy_network_dummy = CartPolePolicy(observation_spec=obs_spec_dummy, action_spec=action_spec_dummy).to(device)
        value_network_dummy = CartPoleValue(observation_spec=obs_spec_dummy).to(device)

    elif args.env_name == 'custom_env':
        from rl_framework.examples.custom_env.custom_environment import CustomComplexEnv
        from rl_framework.examples.custom_env.custom_networks import CustomPolicyNetwork, CustomValueNetwork
        
        # CustomComplexEnv does not have a visual render mode by default
        if args.render:
            print("Warning: CustomComplexEnv does not have a visual rendering mode. Running without rendering.")
            args.render = False

        env_factory = lambda: CustomComplexEnv(device=str(device))
        env = env_factory() # Create one instance for specs and later use

        obs_spec_dummy = env.observation_spec().to(device)
        action_spec_dummy = env.action_spec().to(device)
        
        policy_network_dummy = CustomPolicyNetwork(observation_spec=obs_spec_dummy, action_spec=action_spec_dummy).to(device)
        value_network_dummy = CustomValueNetwork(observation_spec=obs_spec_dummy).to(device)
    else:
        raise ValueError(f"Unknown environment: {args.env_name}")

    # --- Load PPOAgent ---
    # PPOAgent.load_from_checkpoint will instantiate the agent, load state_dict, and use provided networks.
    # The hparams from the checkpoint are also loaded.
    agent = PPOAgent.load_from_checkpoint(
        args.checkpoint_path,
        map_location=device,
        env_factory=env_factory, 
        policy_network=policy_network_dummy,
        value_network=value_network_dummy,
        device=str(device) # Pass device to PPOAgent constructor
    )
    agent.to(device) # Ensure agent itself is on the correct device
    agent.eval() # Set agent to evaluation mode
    if hasattr(agent, 'policy_network') and agent.policy_network is not None:
        agent.policy_network.eval() # Set policy network to evaluation mode
    else:
        print("Warning: Agent does not have a policy_network attribute or it's None after loading.")


    print(f"\nRunning inference for {args.num_episodes} episodes on {args.env_name}...\n")

    for episode in range(args.num_episodes):
        if args.seed is not None:
            # For environments like CartPole, seeding per episode can be useful for consistent starts
            # The seed for reset is environment specific.
            obs = env.reset(seed=args.seed + episode)
        else:
            obs = env.reset()
            
        # Ensure obs is a TensorDict for custom_env, or a tensor for cartpole
        if args.env_name == 'custom_env' and not isinstance(obs, TensorDict):
            # This should not happen if env.reset() returns a TensorDict as designed
            obs = TensorDict(obs, batch_size=[], device=device)
        elif args.env_name == 'cartpole' and isinstance(obs, TensorDict):
            obs = obs.get("obs") # Assuming TensorDict might wrap it

        terminated = False
        truncated = False
        total_reward_episode = 0
        ep_len = 0

        while not (terminated or truncated):
            if args.render and hasattr(env, 'render'):
                env.render()
                if args.render_delay > 0:
                    time.sleep(args.render_delay)

            # Prepare observation for policy
            if isinstance(obs, TensorDict): # CustomEnv
                # Ensure all tensors in TensorDict are on the correct device and have batch dim
                obs_processed = obs.apply(lambda x: x.to(device).unsqueeze(0) if x.ndim == obs_spec_dummy[x.name].domain_ndim else x.to(device), batch_size=[1])
            else: # CartPole
                obs_processed = obs.to(device).unsqueeze(0) # Add batch dimension

            with torch.no_grad():
                # The policy network directly returns the distribution or TensorDict of distributions
                action_dist_or_td = agent.policy_network(obs_processed)

            if isinstance(action_dist_or_td, TensorDict): # CustomPolicyNetwork output
                # For composite actions, take the mode (greedy action) of each component
                action_td = action_dist_or_td.apply(lambda dist: dist.mode, batch_size=[1])
                # The environment step function expects a TensorDict with actions (not distributions)
                # and without the batch dimension for a single step.
                action_to_env = action_td.squeeze(0) 
            else: # CartPolePolicy output (single distribution)
                action_dist = action_dist_or_td
                action = action_dist.mode # Get greedy action
                action_to_env = action.squeeze(0) # Remove batch dim for env
            
            next_obs, reward, terminated, truncated, info = env.step(action_to_env)
            
            total_reward_episode += reward
            ep_len +=1
            obs = next_obs

        print(f"Episode {episode + 1}: Total Reward = {total_reward_episode:.2f}, Length = {ep_len}")

    env.close()
    print("\nInference complete.")

if __name__ == '__main__':
    main()
