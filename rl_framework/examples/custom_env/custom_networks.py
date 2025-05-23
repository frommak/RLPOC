import torch
import torch.nn as nn
from torchrl.data import CompositeSpec, TensorDict, TensorSpec
from torchrl.modules import MLP, ConvNet, ProbabilisticActor, NormalParamWrapper, TanhNormal
from torchrl.modules.distributions import CategoricalDistribution, DiagNormal
from rl_framework.framework.base_policy import BasePolicyNetwork
from rl_framework.framework.base_value import BaseValueNetwork
from typing import Any, cast

class CustomPolicyNetwork(BasePolicyNetwork):
    """
    A policy network for the CustomComplexEnv.
    Processes dictionary observations (image and vector) and outputs
    a composite action distribution (discrete and continuous).
    """
    def __init__(self, observation_spec: CompositeSpec, action_spec: CompositeSpec, hidden_sizes_cnn: list[int] = None, hidden_sizes_mlp: list[int] = None):
        super().__init__()

        if hidden_sizes_cnn is None:
            hidden_sizes_cnn = [16, 32] # num_output_channels for CNN layers
        if hidden_sizes_mlp is None:
            hidden_sizes_mlp = [64, 64]

        self.observation_spec = observation_spec
        self.action_spec = action_spec

        # --- Image Processing Path ---
        image_spec = cast(TensorSpec, observation_spec["image"])
        # Assuming image_spec.shape is (C, H, W) e.g. (3, 16, 16)
        self.cnn = ConvNet(
            in_features=image_spec.shape[0], # num_input_channels
            num_cells=hidden_sizes_cnn, # output channels for each conv layer
            kernel_sizes=[3, 3],
            strides=[1, 1],
            paddings=[1,1],
            activation_class=nn.ReLU,
            # ConvNet will flatten the output
        )
        # Calculate CNN output size (this is a bit manual, TorchRL ConvNet has .out_features after a forward pass with dummy data)
        # For a (3,16,16) input, with 2 layers (3x3 kernel, stride 1, padding 1), output spatial dim remains 16x16.
        # So, cnn_output_features = hidden_sizes_cnn[-1] * image_spec.shape[1] * image_spec.shape[2]
        # However, TorchRL's ConvNet automatically flattens and provides `out_features`
        # Let's create a dummy forward pass to get the out_features
        with torch.no_grad():
            dummy_image = image_spec.rand()
            cnn_output_dim = self.cnn(dummy_image).shape[-1]


        # --- Vector Processing Path ---
        vector_spec = cast(TensorSpec, observation_spec["vector"])
        self.vector_mlp = MLP(
            in_features=vector_spec.shape[-1],
            num_cells=hidden_sizes_mlp,
            out_features=hidden_sizes_mlp[-1],
            activation_class=nn.ReLU,
            activate_last_layer=True
        )
        
        # --- Combined Features Path ---
        combined_features_dim = cnn_output_dim + hidden_sizes_mlp[-1]
        
        # --- Action Heads ---
        # 1. Discrete Action Head
        discrete_action_spec = cast(TensorSpec, action_spec["discrete_action"])
        self.discrete_action_head = MLP(
            in_features=combined_features_dim,
            num_cells=[hidden_sizes_mlp[-1] // 2], # Smaller MLP for head
            out_features=discrete_action_spec.space.n, # Number of discrete choices
            activation_class=nn.ReLU
        )
        
        # 2. Continuous Action Head
        continuous_action_spec = cast(TensorSpec, action_spec["continuous_action"])
        continuous_action_dim = continuous_action_spec.shape[-1]
        
        # MLP to produce parameters for the Normal distribution (mean and std)
        self.continuous_action_params_head = MLP(
            in_features=combined_features_dim,
            num_cells=[hidden_sizes_mlp[-1] // 2],
            out_features=2 * continuous_action_dim, # mean + log_std_dev
            activation_class=nn.ReLU
        )
        # We use NormalParamWrapper to split the output into mean and std
        # and TanhNormal for bounded continuous actions (if needed, here assuming unbounded for simplicity or handled by environment)
        # For simplicity, let's output mean and std for a DiagNormal distribution.
        # TorchRL's ProbabilisticActor often wraps these. Here, we'll return the distributions directly.

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
        discrete_logits = self.discrete_action_head(combined_features)
        discrete_dist = CategoricalDistribution(logits=discrete_logits)
        
        # Get continuous action parameters (mean and log_std) and create distribution
        continuous_params = self.continuous_action_params_head(combined_features)
        continuous_mean, continuous_log_std = continuous_params.chunk(2, dim=-1)
        # Ensure std is positive
        continuous_std = torch.exp(continuous_log_std) 
        continuous_dist = DiagNormal(loc=continuous_mean, scale=continuous_std)
        
        # Return a TensorDict of distributions
        # The keys must match the action_spec for the PPO agent to correctly sample and get log_probs
        return TensorDict({
            "discrete_action": discrete_dist,
            "continuous_action": continuous_dist
        }, batch_size=combined_features.shape[0])


class CustomValueNetwork(BaseValueNetwork):
    """
    A value network for the CustomComplexEnv.
    Processes dictionary observations (image and vector) and outputs a single state value.
    """
    def __init__(self, observation_spec: CompositeSpec, hidden_sizes_cnn: list[int] = None, hidden_sizes_mlp: list[int] = None):
        super().__init__()
        if hidden_sizes_cnn is None:
            hidden_sizes_cnn = [16, 32]
        if hidden_sizes_mlp is None:
            hidden_sizes_mlp = [64, 64]

        self.observation_spec = observation_spec

        # --- Image Processing Path (same as policy) ---
        image_spec = cast(TensorSpec, observation_spec["image"])
        self.cnn = ConvNet(
            in_features=image_spec.shape[0],
            num_cells=hidden_sizes_cnn,
            kernel_sizes=[3, 3], strides=[1,1], paddings=[1,1],
            activation_class=nn.ReLU,
        )
        with torch.no_grad():
            dummy_image = image_spec.rand()
            cnn_output_dim = self.cnn(dummy_image).shape[-1]

        # --- Vector Processing Path (same as policy) ---
        vector_spec = cast(TensorSpec, observation_spec["vector"])
        self.vector_mlp = MLP(
            in_features=vector_spec.shape[-1],
            num_cells=hidden_sizes_mlp,
            out_features=hidden_sizes_mlp[-1],
            activation_class=nn.ReLU,
            activate_last_layer=True
        )
        
        # --- Combined Features Path ---
        combined_features_dim = cnn_output_dim + hidden_sizes_mlp[-1]
        
        # --- Value Head ---
        self.value_head = MLP(
            in_features=combined_features_dim,
            num_cells=[hidden_sizes_mlp[-1] // 2], # Smaller MLP for head
            out_features=1, # Single value output
            activation_class=nn.ReLU,
            activate_last_layer=False # No activation on final value
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
    env = CustomComplexEnv(device="cpu")
    obs_spec = env.observation_spec()
    action_spec = env.action_spec()

    print("Observation Spec from Env:", obs_spec)
    print("Action Spec from Env:", action_spec)

    # Test Policy Network
    policy_net = CustomPolicyNetwork(observation_spec=obs_spec, action_spec=action_spec)
    print("\nCustom Policy Network created.")
    
    # Get a dummy observation TensorDict from the environment
    dummy_obs_td = obs_spec.rand() # Creates a TensorDict with random data matching spec
    if dummy_obs_td.batch_size == torch.Size([]): # if single instance, unsqueeze for batch
        dummy_obs_td = dummy_obs_td.unsqueeze(0)
    
    print("\nDummy Observation TensorDict (batch_size=1):")
    dummy_obs_td.info()

    action_dist_td = policy_net(dummy_obs_td)
    print("\nAction Distribution TensorDict from Policy Network:")
    action_dist_td.info()
    
    # Sample from the distributions in the TensorDict
    # TorchRL's PPO agent would expect to sample from this structure
    # For example, if action_spec is CompositeSpec({"discrete": spec1, "continuous": spec2})
    # then action_dist_td should contain {"discrete": Dist1, "continuous": Dist2}
    # And sampling would be like:
    # sampled_actions_td = action_dist_td.sample() # This works if action_dist_td itself is a Distribution
    # Or, more likely, the PPO agent will iterate through sub-distributions:
    
    sampled_actions = {}
    log_probs = {}

    # The PPOAgent's collector will likely expect the policy's forward pass to return
    # a Distribution that can be sampled, and from which log_probs can be obtained.
    # If the policy returns a TensorDict of distributions, the collector/agent needs to handle this.
    # A common way is to wrap the policy in a ProbabilisticActor from TorchRL,
    # which takes care of constructing a single distribution (often a CompositeDistribution)
    # from the network's output and the action_spec.

    # For this example, let's manually sample from each component:
    print("\nManually sampling actions from distributions:")
    for key, dist in action_dist_td.items():
        sampled_actions[key] = dist.sample()
        log_probs[key] = dist.log_prob(sampled_actions[key]) # Log prob of the sampled action
        print(f"  {key}: Sampled: {sampled_actions[key]}, LogProb: {log_probs[key]}")

    # Create a TensorDict of sampled actions (as PPOAgent might do)
    sampled_actions_td = TensorDict(sampled_actions, batch_size=dummy_obs_td.batch_size)
    print("\nSampled Actions TensorDict:")
    sampled_actions_td.info()


    # Test Value Network
    value_net = CustomValueNetwork(observation_spec=obs_spec)
    print("\nCustom Value Network created.")
    state_values = value_net(dummy_obs_td)
    print("\nState Values from Value Network:", state_values)
    print("State Values Shape:", state_values.shape)

    # Test with single observation (needs to be a TensorDict)
    single_obs_td = obs_spec.rand() # batch_size=[]
    print("\nSingle Observation TensorDict:")
    single_obs_td.info()
    
    single_action_dist_td = policy_net(single_obs_td.unsqueeze(0)) # Policy expects batch
    print("\nSingle Action Distribution TensorDict (after unsqueeze):")
    single_action_dist_td.info()
    
    single_value = value_net(single_obs_td.unsqueeze(0)) # Value net expects batch
    print("\nSingle State Value (after unsqueeze):", single_value)

    print("\nAll example checks passed for custom_networks.py.")
