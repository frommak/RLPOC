import torch
import torch.nn as nn
from torchrl.data import CompositeSpec, TensorSpec
from tensordict import TensorDict
from torchrl.modules.models import MLP, ConvNet 
from torchrl.modules import NormalParamWrapper 
from torchrl.modules.distributions.discrete import OneHotCategorical # Changed from CategoricalDistribution
from rl_framework.framework.base_policy import BasePolicyNetwork
from rl_framework.framework.base_value import BaseValueNetwork
from typing import Any, cast

class CustomPolicyNetwork(BasePolicyNetwork):
    """
    A policy network for the CustomComplexEnv.
    Processes dictionary observations (image and vector) and outputs
    a composite action distribution (discrete and continuous).
    """
    def __init__(self, observation_spec: CompositeSpec, action_spec: CompositeSpec, hidden_sizes_cnn: list[int] = None, hidden_sizes_mlp: list[int] = None, device=None):
        super().__init__()

        if hidden_sizes_cnn is None:
            hidden_sizes_cnn = [16, 32] 
        if hidden_sizes_mlp is None:
            hidden_sizes_mlp = [64, 64]

        self.observation_spec = observation_spec
        self.action_spec = action_spec

        # --- Image Processing Path ---
        image_spec = cast(TensorSpec, observation_spec["image"])
        # Adjust ConvNet depth based on hidden_sizes_cnn length
        cnn_depth = len(hidden_sizes_cnn)
        self.cnn = ConvNet(
            in_features=image_spec.shape[0], 
            num_cells=hidden_sizes_cnn, 
            kernel_sizes=[3] * cnn_depth, # kernel_sizes must match depth
            strides=[1] * cnn_depth,   # strides must match depth
            paddings=[1] * cnn_depth,  # paddings must match depth
            depth=cnn_depth,           # Explicitly set depth
            activation_class=nn.ReLU,
            device=device 
        )
        with torch.no_grad():
            dummy_image_data = image_spec.rand()
            if device:
                dummy_image_data = dummy_image_data.to(device)
            cnn_output_dim = self.cnn(dummy_image_data).shape[-1]

        # --- Vector Processing Path ---
        vector_spec = cast(TensorSpec, observation_spec["vector"])
        self.vector_mlp = MLP(
            in_features=vector_spec.shape[-1],
            num_cells=hidden_sizes_mlp,
            out_features=hidden_sizes_mlp[-1],
            activation_class=nn.ReLU,
            activate_last_layer=True,
            device=device
        )
        
        # --- Combined Features Path ---
        combined_features_dim = cnn_output_dim + hidden_sizes_mlp[-1]
        
        # --- Action Heads ---
        # 1. Discrete Action Head
        discrete_action_spec = cast(TensorSpec, action_spec["discrete_action"])
        self.discrete_action_mlp = MLP( 
            in_features=combined_features_dim,
            num_cells=[hidden_sizes_mlp[-1] // 2], 
            out_features=discrete_action_spec.space.n, 
            activation_class=nn.ReLU,
            device=device
        )
        
        # 2. Continuous Action Head (using NormalParamWrapper)
        continuous_action_spec = cast(TensorSpec, action_spec["continuous_action"])
        continuous_action_dim = continuous_action_spec.shape[-1]
        
        # This MLP will output parameters for the Normal distribution (loc and scale)
        # The output size should be 2 * continuous_action_dim for loc and scale
        self.continuous_action_param_net = MLP(
            in_features=combined_features_dim,
            num_cells=[hidden_sizes_mlp[-1] // 2], 
            out_features=2 * continuous_action_dim, # loc + scale
            activation_class=nn.ReLU, # Activation on hidden layers, but usually not on the final output layer for parameters
            device=device
        )
        # Wrap the MLP with NormalParamWrapper to create the distribution head
        self.continuous_action_head = NormalParamWrapper(
            module=self.continuous_action_param_net, # The network that outputs parameters
            scale_mapping="exp",         # How to transform the scale parameter (e.g., "exp", "softplus")
            scale_lb=0.01,               # Lower bound for the scale (as per prompt, was 1e-4)
            event_ndims=1                # Number of dimensions that constitute an event
                                         # For a DiagNormal, this is 1 (each dim of action is an independent event)
        )
        # NormalParamWrapper will produce a DiagNormal distribution by default.

    def forward(self, observation_td: TensorDict) -> TensorDict:
        """
        Defines the forward pass of the policy network.
        Args:
            observation_td (TensorDict): A TensorDict containing 'image' and 'vector' observations.
        Returns:
            TensorDict: A TensorDict containing action distributions, keyed as in action_spec.
                        e.g., {"discrete_action": CategoricalDistribution, "continuous_action": DiagNormal}
        """
        # Process image
        image_obs = observation_td.get("image")
        if image_obs.ndim == 3: # Add batch dimension if single observation C,H,W
            image_obs = image_obs.unsqueeze(0)
        image_features = self.cnn(image_obs)

        # Process vector
        vector_obs = observation_td.get("vector")
        if vector_obs.ndim == 1: # Add batch dimension if single observation D
            vector_obs = vector_obs.unsqueeze(0)
        vector_features = self.vector_mlp(vector_obs)
        
        # Combine features
        combined_features = torch.cat([image_features, vector_features], dim=-1)
        
        # Get discrete action logits and create distribution
        discrete_logits = self.discrete_action_mlp(combined_features)
        discrete_dist = OneHotCategorical(logits=discrete_logits) # Changed to OneHotCategorical
        
        # Get continuous action distribution using NormalParamWrapper
        continuous_action_dist = self.continuous_action_head(combined_features)
        
        # Return a TensorDict of distributions
        # The keys must match the action_spec for the PPO agent to correctly sample and get log_probs
        return TensorDict({
            "discrete_action": discrete_dist,
            "continuous_action": continuous_action_dist # This is now a DiagNormal distribution
        }, batch_size=combined_features.shape[0])


class CustomValueNetwork(BaseValueNetwork):
    """
    A value network for the CustomComplexEnv.
    Processes dictionary observations (image and vector) and outputs a single state value.
    """
    def __init__(self, observation_spec: CompositeSpec, hidden_sizes_cnn: list[int] = None, hidden_sizes_mlp: list[int] = None, device=None): # Added device
        super().__init__()
        if hidden_sizes_cnn is None:
            hidden_sizes_cnn = [16, 32]
        if hidden_sizes_mlp is None:
            hidden_sizes_mlp = [64, 64]

        self.observation_spec = observation_spec

        # --- Image Processing Path (same as policy) ---
        image_spec = cast(TensorSpec, observation_spec["image"])
        # Adjust ConvNet depth based on hidden_sizes_cnn length
        cnn_depth_val = len(hidden_sizes_cnn)
        self.cnn = ConvNet(
            in_features=image_spec.shape[0],
            num_cells=hidden_sizes_cnn,
            kernel_sizes=[3] * cnn_depth_val, 
            strides=[1] * cnn_depth_val,   
            paddings=[1] * cnn_depth_val,  
            depth=cnn_depth_val,          
            activation_class=nn.ReLU,
            device=device
        )
        with torch.no_grad():
            dummy_image = image_spec.rand().to(device) if device else image_spec.rand()
            cnn_output_dim = self.cnn(dummy_image).shape[-1]

        # --- Vector Processing Path (same as policy) ---
        vector_spec = cast(TensorSpec, observation_spec["vector"])
        self.vector_mlp = MLP(
            in_features=vector_spec.shape[-1],
            num_cells=hidden_sizes_mlp,
            out_features=hidden_sizes_mlp[-1],
            activation_class=nn.ReLU,
            activate_last_layer=True,
            device=device
        )
        
        # --- Combined Features Path ---
        combined_features_dim = cnn_output_dim + hidden_sizes_mlp[-1]
        
        # --- Value Head ---
        self.value_head = MLP(
            in_features=combined_features_dim,
            num_cells=[hidden_sizes_mlp[-1] // 2], 
            out_features=1, 
            activation_class=nn.ReLU,
            activate_last_layer=False,
            device=device
        )

    def forward(self, observation_td: TensorDict) -> torch.Tensor:
        """
        Defines the forward pass of the value network.
        Args:
            observation_td (TensorDict): A TensorDict containing 'image' and 'vector' observations.
        Returns:
            torch.Tensor: A tensor representing the state value. Shape: [batch_size, 1].
        """
        # Process image
        image_obs = observation_td.get("image")
        if image_obs.ndim == 3: # Add batch dimension
            image_obs = image_obs.unsqueeze(0)
        image_features = self.cnn(image_obs)

        # Process vector
        vector_obs = observation_td.get("vector")
        if vector_obs.ndim == 1: # Add batch dimension
            vector_obs = vector_obs.unsqueeze(0)
        vector_features = self.vector_mlp(vector_obs)
        
        # Combine features
        combined_features = torch.cat([image_features, vector_features], dim=-1)
        
        # Get state value
        value = self.value_head(combined_features)
        return value

if __name__ == '__main__':
    from rl_framework.examples.custom_env.custom_environment import CustomComplexEnv
    
    # Use the actual environment to get specs
    device_for_test = "cpu"
    env = CustomComplexEnv(device=device_for_test)
    obs_spec = env.observation_spec() # Already on device_for_test from env
    action_spec = env.action_spec() # Already on device_for_test from env

    print("Observation Spec from Env:", obs_spec)
    print("Action Spec from Env:", action_spec)

    # Test Policy Network
    policy_net = CustomPolicyNetwork(observation_spec=obs_spec, action_spec=action_spec, device=device_for_test)
    print("\nCustom Policy Network created.")
    
    # Get a dummy observation TensorDict from the environment
    dummy_obs_td = obs_spec.rand() 
    if dummy_obs_td.batch_size == torch.Size([]): 
        dummy_obs_td = dummy_obs_td.unsqueeze(0)
    
    print("\nDummy Observation TensorDict (batch_size=1):")
    dummy_obs_td.info()

    action_dist_td = policy_net(dummy_obs_td)
    print("\nAction Distribution TensorDict from Policy Network:")
    action_dist_td.info()
        
    sampled_actions = {}
    log_probs = {}

    print("\nManually sampling actions from distributions:")
    for key, dist_val in action_dist_td.items(): 
        sampled_actions[key] = dist_val.sample()
        # OneHotCategorical log_prob expects one-hot encoded actions. 
        # For testing, we might need to convert sampled_actions[key] if it's not already one-hot.
        # However, the PPO agent handles sampling and log_prob calculation internally based on policy output.
        # For this simple test, we'll just show the sample.
        # log_probs[key] = dist_val.log_prob(sampled_actions[key]) 
        # print(f"  {key}: Sampled: {sampled_actions[key]}, LogProb: {log_probs[key].shape if hasattr(log_probs[key], 'shape') else log_probs[key]}")
        print(f"  {key}: Sampled: {sampled_actions[key]}")


    sampled_actions_td = TensorDict(sampled_actions, batch_size=dummy_obs_td.batch_size, device=device_for_test) 
    print("\nSampled Actions TensorDict:")
    sampled_actions_td.info()


    # Test Value Network
    value_net = CustomValueNetwork(observation_spec=obs_spec, device=device_for_test)
    print("\nCustom Value Network created.")
    state_values = value_net(dummy_obs_td)
    print("\nState Values from Value Network:", state_values)
    print("State Values Shape:", state_values.shape)

    # Test with single observation (needs to be a TensorDict)
    single_obs_td = obs_spec.rand() 
    print("\nSingle Observation TensorDict:")
    single_obs_td.info()
    
    # Policy and value nets expect batched input, even if batch size is 1
    single_action_dist_td = policy_net(single_obs_td.unsqueeze(0)) 
    print("\nSingle Action Distribution TensorDict (after unsqueeze):")
    single_action_dist_td.info()
    
    single_value = value_net(single_obs_td.unsqueeze(0)) 
    print("\nSingle State Value (after unsqueeze):", single_value)

    print("\nAll example checks passed for custom_networks.py.")
