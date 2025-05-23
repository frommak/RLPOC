import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage, TensorDict # Adjusted import for TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
import lightning as L

from .base_environment import BaseEnvironment
from .base_policy import BasePolicyNetwork
from .base_value import BaseValueNetwork

class PPOAgent(L.LightningModule):
    def __init__(
        self,
        env_factory: callable, # A function that returns an instance of BaseEnvironment
        policy_network: BasePolicyNetwork,
        value_network: BaseValueNetwork,
        lr_policy: float = 3e-4,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        ppo_epochs: int = 10,
        num_rollout_steps: int = 2048, # Steps per environment before update
        batch_size: int = 64, # Minibatch size for PPO updates
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = 'cpu', # or 'cuda' if available
        # Add other necessary PPO parameters
    ):
        super().__init__()
        # Using logger=False to avoid issues with saving hyperparameters that are modules or callables
        self.save_hyperparameters(logger=False, ignore=['env_factory', 'policy_network', 'value_network'])

        self.env_factory = env_factory
        self.policy_network = policy_network
        self.value_network = value_network
        self.device_ = torch.device(device) # Renamed to avoid conflict with Lightning's self.device

        # Initialize environment for specs, then close it.
        temp_env = self.env_factory()
        self.obs_spec = temp_env.observation_spec()
        self.action_spec = temp_env.action_spec()
        temp_env.close()

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.num_rollout_steps = num_rollout_steps
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        
        # Move networks to device
        self.policy_network.to(self.device_)
        self.value_network.to(self.device_)

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.num_rollout_steps),
            batch_size=self.batch_size 
        )
        
        self.collector = SyncDataCollector(
            create_env_fn=self.env_factory,
            policy=self.policy_network, 
            frames_per_batch=self.num_rollout_steps,
            total_frames=-1, 
            device=self.device_,
            storing_device=self.device_ 
        )

    def configure_optimizers(self):
        policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.hparams.lr_policy)
        value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.hparams.lr_value)
        return [policy_optimizer, value_optimizer]

    def training_step(self, batch, batch_idx):
        self.policy_network.train()
        self.value_network.train()
        
        # Set exploration type for policy during collection if needed
        # This ensures that the policy explores rather than choosing the greedy action
        with set_exploration_type(ExplorationType.RANDOM):
            collected_data = self.collector.next()
        
        collected_data = collected_data.to(self.device_)
        
        advantages, returns = self._calculate_advantages_and_returns(collected_data)

        collected_data["advantages"] = advantages
        collected_data["returns"] = returns
        
        # Initialize losses to a default value (e.g., 0 or NaN)
        # This is to ensure they are defined in case the inner loop doesn't run (e.g. if num_samples < batch_size)
        policy_loss = torch.tensor(0.0, device=self.device_)
        value_loss = torch.tensor(0.0, device=self.device_)
        entropy_loss = torch.tensor(0.0, device=self.device_)
        
        num_samples = collected_data.shape[0]
        if num_samples == 0: # Should not happen if collector works
             # Log or handle empty data scenario
            self.log_dict({
                "loss/policy_loss": 0.0,
                "loss/value_loss": 0.0,
                "loss/entropy_loss": 0.0,
                "info/mean_reward_rollout": 0.0,
            }, prog_bar=True)
            return torch.tensor(0.0, device=self.device_, requires_grad=True) # Return a zero loss

        for _ in range(self.ppo_epochs):
            ids = torch.randperm(num_samples, device=self.device_) # Ensure ids are on the same device
            
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                if end > num_samples: 
                    # if num_samples % self.batch_size != 0: # if we want to process the last partial batch
                    #    minibatch_ids = ids[start:]
                    # else:
                    continue # Skip partial last batch for simplicity
                
                minibatch_ids = ids[start:end]
                minibatch = collected_data[minibatch_ids]

                obs = minibatch["obs"]
                actions = minibatch["action"]
                # Ensure old_log_probs are on the correct device and shape
                old_log_probs = minibatch["sample_log_prob"].to(self.device_)
                if old_log_probs.ndim > 1 and old_log_probs.shape[-1] == 1 :
                    old_log_probs = old_log_probs.squeeze(-1)

                advantages_mb = minibatch["advantages"]
                returns_mb = minibatch["returns"]
                
                current_policy_dist = self.policy_network(obs)
                new_log_probs = current_policy_dist.log_prob(actions)
                if new_log_probs.ndim > 1 and new_log_probs.shape[-1] == 1 :
                    new_log_probs = new_log_probs.squeeze(-1)

                ratio = (new_log_probs - old_log_probs).exp()

                surr1 = ratio * advantages_mb
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_mb
                policy_loss = -torch.min(surr1, surr2).mean()
                
                entropy_dist = current_policy_dist.entropy()
                # Some distributions might return entropy per event, others sum it. Mean is usually safe.
                if entropy_dist.ndim > 1 and entropy_dist.shape[-1] ==1: # Ensure it's [batch_size]
                    entropy_dist = entropy_dist.squeeze(-1)
                entropy_loss = -entropy_dist.mean()
                
                total_policy_loss = policy_loss + self.entropy_coef * entropy_loss

                current_values = self.value_network(obs)
                value_loss = (current_values - returns_mb).pow(2).mean()
                
                total_value_loss = self.value_loss_coef * value_loss

                opt_policy, opt_value = self.optimizers()
                
                opt_policy.zero_grad()
                self.manual_backward(total_policy_loss)
                if self.max_grad_norm is not None:
                    clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
                opt_policy.step()

                opt_value.zero_grad()
                self.manual_backward(total_value_loss)
                if self.max_grad_norm is not None:
                    clip_grad_norm_(self.value_network.parameters(), self.max_grad_norm)
                opt_value.step()
        
        mean_rollout_reward = collected_data["next"]["reward"].mean() # Typically use "next_reward" or "reward" from the done steps

        self.log_dict({
            "loss/policy_loss": policy_loss,
            "loss/value_loss": value_loss,
            "loss/entropy_loss": entropy_loss,
            "info/mean_reward_rollout": mean_rollout_reward,
        }, prog_bar=True, logger=True)
        
        return total_policy_loss + total_value_loss

    def _calculate_advantages_and_returns(self, rollout_data: TensorDict):
        rewards = rollout_data["next"]["reward"] # TorchRL collectors store reward for (s,a,s') transitions with s'
        dones = rollout_data["next"]["done"]    # Same for dones

        with torch.no_grad():
            values = self.value_network(rollout_data["obs"]) 

        # Get value for the state after the last action in the rollout
        # The 'next' tensordict contains 'obs' which is s_t+1
        last_next_obs = rollout_data["next"]["obs"][-1:] 
        with torch.no_grad():
            next_value = self.value_network(last_next_obs)

        advantages = torch.zeros_like(rewards, device=self.device_) # ensure advantages are on the correct device
        last_gae_lam = 0
        
        # Squeeze if necessary, ensure they are on the correct device
        rewards = rewards.to(self.device_).squeeze(-1) if rewards.ndim > 1 and rewards.shape[-1] == 1 else rewards.to(self.device_)
        dones = dones.to(self.device_).squeeze(-1) if dones.ndim > 1 and dones.shape[-1] == 1 else dones.to(self.device_)
        values = values.to(self.device_).squeeze(-1) if values.ndim > 1 and values.shape[-1] == 1 else values.to(self.device_)
        next_value = next_value.to(self.device_).squeeze(-1) if next_value.ndim > 1 and next_value.shape[-1] == 1 else next_value.to(self.device_)


        for t in reversed(range(self.num_rollout_steps)):
            if t == self.num_rollout_steps - 1:
                next_non_terminal = 1.0 - dones[t].float() # convert bool to float
                next_val_for_delta = next_value
            else:
                next_non_terminal = 1.0 - dones[t].float()
                next_val_for_delta = values[t+1]
            
            delta = rewards[t] + self.gamma * next_val_for_delta * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        
        returns = advantages + values 
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages.unsqueeze(-1), returns.unsqueeze(-1)

    def on_train_start(self):
        # Ensure networks are on the correct device (self.device is Lightning's managed device)
        self.policy_network.to(self.device)
        self.value_network.to(self.device)
        
        # Update self.device_ to match Lightning's self.device if they differ
        # This is important if the 'device' PPOAgent was initialized with
        # is different from what the Lightning Trainer decides (e.g. based on accelerator availability)
        if self.device_ != self.device:
            print(f"PPOAgent device_ {self.device_} updated to Trainer device {self.device}.")
            self.device_ = self.device
            # Collector's device might also need an update.
            # This is complex as collector is stateful.
            # For now, we assume initial device matches or manual restart/re-init of collector if device changes post-init.
            # A safer approach is to initialize collector in setup("fit") where self.device is known.
            # For this subtask, we stick to the provided structure.
            # If collector needs device update, it should be re-initialized or its policy device updated.
            if hasattr(self.collector, 'policy'):
                 if hasattr(self.collector.policy, 'to'):
                      self.collector.policy.to(self.device_) # if policy is a module
            if hasattr(self.collector, 'env_constructors') and self.collector.env_constructors:
                 if hasattr(self.collector.env_constructors[0], 'device'): # Heuristic
                      self.collector.env_constructors[0].device = self.device_

    # We need to ensure that the LightningModule has `automatic_optimization = False`
    # because we are calling `self.manual_backward()` and `optimizer.step()` ourselves.
    @property
    def automatic_optimization(self) -> bool:
        return False

```
