from enviorment import Enviorment
from actorCritic import ActorCritic
import torch
import torch.nn as nn
from device import device

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space  # Flag indicating action space type (continuous or discrete)

        if has_continuous_action_space:
            self.action_std = action_std_init  # Initial action standard deviation (for continuous actions)

        self.gamma = gamma  # Discount factor for future rewards
        self.eps_clip = eps_clip  # PPO clip parameter for policy update
        self.K_epochs = K_epochs  # Number of epochs (policy update iterations) per PPO step
        
        self.buffer = Enviorment()  # Create an Environment object to store rollouts

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)  # Actor-Critic model
        self.optimizer = torch.optim.Adam([  # Create optimizer for policy and critic updates
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},  # Actor parameters with learning rate (lr_actor)
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}   # Critic parameters with learning rate (lr_critic)
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)  # Copy of the policy (old policy)
        self.policy_old.load_state_dict(self.policy.state_dict())  # Load the current policy weights into the old policy

        self.MseLoss = nn.MSELoss()  # Mean Squared Error loss function for critic updates

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std  # Update action standard deviation
            self.policy.set_action_std(new_action_std)  # Set new std in the current policy
            self.policy_old.set_action_std(new_action_std)  # Set new std in the old policy (for consistency)
        else:
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")  # Informative message for discrete case

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate  # Decay action std
            self.action_std = round(self.action_std, 4)  # Round to 4 decimal places
            if self.action_std <= min_action_std:  # Clip to minimum std
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)  # Update policy and old policy with new std
        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")  # Informative message for discrete case
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):
        """Samples an action from the current policy for exploration.

        Args:
            state (np.ndarray): The current state of the environment.

        Returns:
            np.ndarray: The sampled action.
        """

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, _, _ = self.policy.act(state)  # Only need action here

            return action.detach().cpu().numpy().flatten()  # Detach, move to CPU, and flatten

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, _, _ = self.policy.act(state)  # Only need action here

            return action.item()  # Return the action as a single value (discrete case)
