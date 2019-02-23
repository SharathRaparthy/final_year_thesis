#import neccessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import copy
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import random
import gym
from gym import spaces
import argparse
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.simulator import Simulator
from gym_duckietown.wrappers import UndistortWrapper
# from hyperdash import Experiment
import time
import os
import matplotlib.image as mpimg
from skimage.color import rgb2gray
import pickle
# def rgb2gray(rgb):
#     return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def data_preprocess(image_frames):
    image_frames = image_frames.transpose((1,2,0))
    # print(image_frames.shape)
    gray_image = rgb2gray(image_frames)
    return gray_image
stack_size = 4





# exp = Experiment("duckietown - ddpg- udem1 training")
# exp = Experiment("[duckietown] - ddpg")




#initialise actor network

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_dim, action_dim, max_action):

        super(Actor, self).__init__()
        flat_size = 32 * 9 * 14
        init_wt = 3e-3
        num_filters = 4

        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        self.conv1 = nn.Conv2d(num_filters, 32, 8, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 4, stride=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(.5)

        self.lin1 = nn.Linear(flat_size, 512)
        self.lin1.weight.data = fanin_init(self.lin1.weight.data.size())
        self.lin2 = nn.Linear(512, action_dim)
        self.lin2.weight.data.uniform_(-init_wt,init_wt)

        self.max_action = max_action

    def forward(self, x):
        x = self.bn1(self.lr(self.conv1(x)))
        x = self.bn2(self.lr(self.conv2(x)))
        x = self.bn3(self.lr(self.conv3(x)))
        x = self.bn4(self.lr(self.conv4(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(x)
        x = self.lr(self.lin1(x))
         # because we don't want our duckie to go backwards
        x = self.lin2(x)
        x[:, 0] = self.max_action * self.sigm(x[:, 0])  # because we don't want the duckie to go backwards
        x[:, 1] = self.tanh(x[:, 1])

        return x


#initialise critic network
class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self,state_dim, action_dim, max_action):
        super(Critic, self).__init__()
        flat_size = 32 * 9 * 14
        init_wt = 3e-3
        num_filters = 4

        self.lr = nn.LeakyReLU()

        self.conv1 = nn.Conv2d(num_filters, 32, 8, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 4, stride=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(.5)

        self.lin1 = nn.Linear(flat_size, 256)
        self.lin1.weight.data = fanin_init(self.lin1.weight.data.size())
        self.lin2 = nn.Linear(256 + action_dim, 128)
        self.lin2.weight.data = fanin_init(self.lin2.weight.data.size())
        self.lin3 = nn.Linear(128, 1)
        self.lin3.weight.data.uniform_(-init_wt,init_wt)

    def forward(self, states, actions):
        x = self.bn1(self.lr(self.conv1(states)))
        x = self.bn2(self.lr(self.conv2(x)))
        x = self.bn3(self.lr(self.conv3(x)))
        x = self.bn4(self.lr(self.conv4(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = self.lr(self.lin1(x))
        x = self.lr(self.lin2(torch.cat([x, actions], 1)))  # c
        x = self.lin3(x)
        return x


#noise
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

#replay buffer class
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    def save(self):
        with open("experiences.txt", "wb") as fp:
            pickle.dump(self.memory, fp)
    def open(self):
        with open("experiences.txt", "rb") as fo:
            b = pickle.load(fo)
            self.memory = b
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

#         states, actions, rewards, next_states, dones = np.expand_dims(experience.state, axis=0),np.expand_dims(actions,axis=0),np.expand_dims(reward,axis=0) ,np.expand_dims(next_state,axis=0) ,np.expand_dims(done,axis=0)
        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None],axis=0)).float().to(device)



        actions = torch.from_numpy(np.stack([e.action for e in experiences if e is not None],axis=0)).float().to(device)
        rewards = torch.from_numpy(np.stack([e.reward for e in experiences if e is not None],axis=0)).float().to(device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None],axis=0)).float().to(device)
        dones = torch.from_numpy(np.stack([e.done for e in experiences if e is not None],axis=0).astype(np.uint8)).float().to(device)


        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)




class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size,max_action, random_seed,load):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.load = load

        #else:
        self.actor_local = Actor(state_size, action_size,max_action).to(device)
        self.actor_target = Actor(state_size, action_size, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, max_action).to(device)
        self.critic_target = Critic(state_size, action_size,max_action).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    def save(self, filename, directory):
        torch.save(self.actor_local.state_dict(),'{}/{}_actor.pth'.format(directory, filename))
        torch.save(self.critic_local.state_dict(),'{}/{}_critic.pth'.format(directory, filename))

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)


    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
#         if len(self.memory) <= BATCH_SIZE:
        state = state.unsqueeze(0)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        action[0][0] = np.clip(action[0][0], 0,1)
        action[0][1] = np.clip(action[0][1],-1,1)
        return action[0]

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):


        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        rewards = rewards.unsqueeze(1)

        dones = dones.unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)


        critic_loss = F.mse_loss(Q_expected, Q_targets)


        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



#wrappers
class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(120, 160, 3)):
        super(ResizeWrapper, self).__init__(env)
        self.observation_space.shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            shape,
            dtype=self.observation_space.dtype)
        self.shape = shape

    def observation(self, observation):
        from scipy.misc import imresize
        return imresize(observation, self.shape)


class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)


class ImgWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ImgWrapper, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -5
        # elif reward > 0:
        #     reward *= 10
        # else:
        #     reward = reward -1

        return reward


# this is needed because at max speed the duckie can't turn anymore
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)

    def action(self, action):
        action_ = [action[0] * 0.6, action[1]]
        return action_


env = gym.make('Duckietown-udem1-v0')

env = ResizeWrapper(env)
env = NormalizeWrapper(env)
env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
env = ActionWrapper(env)
env = DtRewardWrapper(env)


def stack_images(stacked_frames, state, new_episode):

    frame = data_preprocess(state)


    stack_size = 4
    if new_episode:
        stacked_frames = deque([torch.zeros(state.shape) for i in range(stack_size)], maxlen=4)#state.shape returns (120,160,3) but we require only 1 channel
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis = 0)
        # print(stacked_state.shape)
    else:
    	stacked_frames.append(frame)
    	stacked_state = np.stack(stacked_frames, axis = 0)
    return stacked_state, stacked_frames
BUFFER_SIZE = 2  # replay buffer size |EXP-4: 20k| EXP5: 20k |
BATCH_SIZE = 16        # minibatch size |EXP-4:128| EXP5: 32 |
GAMMA = 0.99            # discount factor |EXP-4:0.99| EXP5: 0.99 |
TAU = 0.001              # for soft update of target parameters |EXP-4:1e-3| EXP5: 0.001 |
LR_ACTOR = 1e-4         # learning rate of the actor |EXP-4:same| EXP5: same |
LR_CRITIC = 1e-3        # learning rate of the critic |EXP-4:same| EXP5: same |
WEIGHT_DECAY = 1e-2        # L2 weight decay |EXP-4:0| EXP5: 0.001 |

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
state_size = env.observation_space.shape
action_size = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = Agent(state_size=state_size, action_size=action_size,max_action = max_action, random_seed=10,load=True)
if not os.path.exists("./ddpg_models"):
    os.makedirs("./ddpg_models")
def sigmoid(x):

    return 1/(1+np.exp(-x))


name = "checkpoint"
experiment = 13 #13 udem1
filename = "{}_{}".format(name,experiment)
def ddpg():
    total_timesteps = 0
    stacked_frames  =  deque([np.zeros((120,160), dtype=np.int) for i in range(stack_size)], maxlen=4)

    max_timesteps = 1e6
    n_episodes = 0

    scores = []
    score = 0
    start_timesteps = 10000 #|EXP-4:15k| EXP5: 10k |
    episode_reward = 0
    episode_timesteps = 500
    env_timesteps = 0
    done = False
    reward = 0
    eval_every = 10
#     for i_episode in range(1, n_episodes+1):
    state = env.reset()

    save = False

    while total_timesteps < max_timesteps:
        state, stacked_frames = stack_images(stacked_frames,state, True)
        # print(state.shape)

        for e_steps in range(episode_timesteps):
            if done:
                if save:
                    agent.save(filename,directory="./ddpg_models")
                n_episodes += 1
                state = env.reset()
                state, stacked_frames = stack_images(stacked_frames,state, True)

                if n_episodes  % eval_every == 0:
                    avg_reward = score/eval_every
                    scores.append(avg_reward)
                    # exp.metric("Rewards", scores[-1])
                    print('Total TimeSteps: {}, Episode Number: {}, Average Episode Reward: {}'.format(total_timesteps,n_episodes,avg_reward))
                    score = 0


            if total_timesteps < start_timesteps:
                action = env.action_space.sample()
                action[0] = sigmoid(action[0])

            else:
                action = agent.act(state)


            next_state, reward, done, _ = env.step(action)
            next_state, stacked_frames = stack_images(stacked_frames, next_state, False)
            env.render()

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            total_timesteps += 1

    return scores
test = False
if test:
    scores = ddpg()

    scores = np.array(scores)
    np.save('rewards_' + 'udem1_stacked' + '.npy',scores)
