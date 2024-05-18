import os
from datetime import datetime
import gym
import torch
from ppo import PPO
import numpy as np

print("============================================================================================")

################################### Training ###################################

####### Initialize environment hyperparameters ######

env_name = "CartPole-v1"  # Name of the environment
has_continuous_action_space = False  # Whether the action space is continuous

max_ep_len = 400  # Maximum timesteps in one episode
max_training_timesteps = int(1e5)  # Stop training if timesteps exceed this number

# Frequencies for various activities (in terms of timesteps)
print_freq = max_ep_len * 4  # Frequency of printing average reward
log_freq = max_ep_len * 2  # Frequency of logging average reward
save_model_freq = int(2e4)  # Frequency of saving the model

action_std = None  # Standard deviation for action, if needed for continuous actions

#####################################################

# Ensure print/log frequencies are greater than max_ep_len
assert print_freq > max_ep_len
assert log_freq > max_ep_len

################ PPO hyperparameters ################

update_timestep = max_ep_len * 4  # Update policy every n timesteps
K_epochs = 40  # Number of epochs for policy update
eps_clip = 0.2  # Clipping parameter for PPO
gamma = 0.99  # Discount factor for reward

# Learning rates for the actor and critic networks
lr_actor = 0.0003  
lr_critic = 0.001  

random_seed = 0  # Set random seed if required (0 means no random seed)

#####################################################

print("Training environment name: " + env_name)

env = gym.make(env_name)  # Initialize the environment

# State space dimension
state_dim = env.observation_space.shape[0]

# Action space dimension
if has_continuous_action_space:
    action_dim = env.action_space.shape[0]
else:
    action_dim = env.action_space.n

###################### Logging ######################

# Directory for logging
log_dir = "PPO_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Directory for environment-specific logs
log_dir = log_dir + '/' + env_name + '/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Determine the run number by counting existing log files
run_num = len(next(os.walk(log_dir))[2])

# Create a new log file for each run
log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

print("Current logging run number for " + env_name + ": ", run_num)
print("Logging at: " + log_f_name)

#####################################################


################### Checkpointing ###################

# Run number for pre-trained model to prevent overwriting existing weights
run_num_pretrained = 0  # Change this value as needed to avoid overwriting

# Directory for saving pre-trained models
directory = "PPO_preTrained"
if not os.path.exists(directory):
    os.makedirs(directory)

# Subdirectory for the specific environment
directory = directory + '/' + env_name + '/'
if not os.path.exists(directory):
    os.makedirs(directory)

# Path for saving the checkpoint
checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
print("Save checkpoint path: " + checkpoint_path)

#####################################################



############# print all hyperparameters #############

print("--------------------------------------------------------------------------------------------")

print("max training timesteps : ", max_training_timesteps)
print("max timesteps per episode : ", max_ep_len)

print("model saving frequency : " + str(save_model_freq) + " timesteps")
print("log frequency : " + str(log_freq) + " timesteps")
print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")

print("--------------------------------------------------------------------------------------------")

print("state space dimension : ", state_dim)
print("action space dimension : ", action_dim)

print("--------------------------------------------------------------------------------------------")

if has_continuous_action_space:
    print("Initializing a continuous action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("starting std of action distribution : ", action_std)
    print("decay rate of std of action distribution : ", action_std_decay_rate)
    print("minimum std of action distribution : ", min_action_std)
    print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")

else:
    print("Initializing a discrete action space policy")

print("--------------------------------------------------------------------------------------------")

print("PPO update frequency : " + str(update_timestep) + " timesteps") 
print("PPO K epochs : ", K_epochs)
print("PPO epsilon clip : ", eps_clip)
print("discount factor (gamma) : ", gamma)

print("--------------------------------------------------------------------------------------------")

print("optimizer learning rate actor : ", lr_actor)
print("optimizer learning rate critic : ", lr_critic)

if random_seed:
    print("--------------------------------------------------------------------------------------------")
    print("setting random seed to ", random_seed)
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)

#####################################################

print("============================================================================================")

################# Training Procedure ################

# Initialize a PPO agent with the specified hyperparameters
ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

# Track the total training time
start_time = datetime.now().replace(microsecond=0)
print("Started training at (IST): ", start_time)
print("============================================================================================")

# Create and open a log file for writing
log_f = open(log_f_name, "w+")
log_f.write('episode,timestep,reward\n')

# Variables for tracking running rewards and episodes for printing and logging
print_running_reward = 0
print_running_episodes = 0

log_running_reward = 0
log_running_episodes = 0

time_step = 0
i_episode = 0

# Training loop
while time_step <= max_training_timesteps:
    
    state = env.reset()
    current_ep_reward = 0

    for t in range(1, max_ep_len + 1):
        
        # Select action according to policy
        action = ppo_agent.select_action(state)
        state, reward, done, _ = env.step(action)
        
        # Save reward and terminal state
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(done)
        
        time_step += 1
        current_ep_reward += reward

        # Update PPO agent at specified intervals
        if time_step % update_timestep == 0:
            ppo_agent.update()

        # Decay action standard deviation if the action space is continuous
        if has_continuous_action_space and time_step % action_std_decay_freq == 0:
            ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

        # Log average reward at specified intervals
        if time_step % log_freq == 0:
            log_avg_reward = log_running_reward / log_running_episodes
            log_avg_reward = round(log_avg_reward, 4)

            log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
            log_f.flush()

            log_running_reward = 0
            log_running_episodes = 0

        # Print average reward at specified intervals
        if time_step % print_freq == 0:
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = round(print_avg_reward, 2)

            print("Episode: {} \t\t Timestep: {} \t\t Average Reward: {}".format(i_episode, time_step, print_avg_reward))

            print_running_reward = 0
            print_running_episodes = 0
            
        # Save model weights at specified intervals
        if time_step % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            print("Saving model at: " + checkpoint_path)
            ppo_agent.save(checkpoint_path)
            print("Model saved")
            print("Elapsed Time: ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")
            
        # End the episode if done
        if done:
            break

    print_running_reward += current_ep_reward
    print_running_episodes += 1

    log_running_reward += current_ep_reward
    log_running_episodes += 1

    i_episode += 1

log_f.close()
env.close()


# print total training time
print("============================================================================================")
end_time = datetime.now().replace(microsecond=0)
print("Started training at (IST) : ", start_time)
print("Finished training at (IST) : ", end_time)
print("Total training time  : ", end_time - start_time)
print("============================================================================================")