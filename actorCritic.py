import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from device import device

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()  # Initialize the PyTorch module

        self.has_continuous_action_space = has_continuous_action_space  # Flag indicating action space type

        if has_continuous_action_space:
            self.action_dim = action_dim  # Store action dimension (for continuous spaces)
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)  # Initialize action variance (for continuous spaces)

        # Actor network (policy) for choosing actions
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),  # First linear layer with 64 hidden units and Tanh activation
                nn.Tanh(),
                nn.Linear(64, 64),  # Second linear layer with 64 hidden units and Tanh activation
                nn.Tanh(),
                nn.Linear(64, action_dim),  # Output layer with same dimension as action space and Tanh activation (for continuous)
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),  # First linear layer with 64 hidden units and Tanh activation
                nn.Tanh(),
                nn.Linear(64, 64),  # Second linear layer with 64 hidden units and Tanh activation
                nn.Linear(64, action_dim),  # Output layer with same dimension as action space and Softmax activation (for discrete)
                nn.Softmax(dim=-1)  # Softmax for probability distribution over discrete actions
            )

        # Critic network (value function) for estimating state values
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),  # First linear layer with 64 hidden units and Tanh activation
            nn.Tanh(),
            nn.Linear(64, 64),  # Second linear layer with 64 hidden units and Tanh activation
            nn.Linear(64, 1)  # Output layer with 1 unit for the estimated state value
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)  # Update action variance
        else:
            print("calling set_action_std() - Not applicable for discrete actions")  # Informative message for discrete case

    def forward(self):
        raise NotImplementedError  # This method is intentionally left unimplemented, likely meant for inheritance

    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)  # Get action mean from the actor network
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)  # Create diagonal covariance matrix from action variance
            dist = MultivariateNormal(action_mean, cov_mat)  # Create a multivariate normal distribution for continuous actions
        else:
            action_probs = self.actor(state)  # Get action probabilities from the actor network
            dist = Categorical(action_probs)  # Create a categorical distribution for discrete actions

        action = dist.sample()  # Sample an action from the distribution
        action_logprob = dist.log_prob(action).detach()  # Calculate the log probability of the sampled action and detach from computational graph
        state_val = self.critic(state).detach()  # Get the state value from the critic network and detach from computational graph

        return action, action_logprob, state_val  # Return the sampled action, its log probability, and the state value

    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            # Calculate action mean
            action_mean = self.actor(state)

            # Expand action variance
            action_var = self.action_var.expand_as(action_mean)

            # Create covariance matrix
            cov_mat = torch.diag_embed(action_var).to(device)

            # Create multivariate normal distribution
            dist = MultivariateNormal(action_mean, cov_mat)

            # Handle single-dimensional action case (optional)
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            # Discrete case (unchanged)
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        # Calculate action log probabilities
        action_logprobs = dist.log_prob(action)

        # Calculate distribution entropy
        dist_entropy = dist.entropy()

        # Get state values
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy
