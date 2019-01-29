import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
from skimage.color import rgb2gray
from skimage import transform
import torch.nn.functional as F
import gym
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pandas as pd
from itertools import count
from collections import deque
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.simulator import Simulator
from gym_duckietown.wrappers import UndistortWrapper
import argparse

#args
env = gym.make('Duckietown-udem1-v0')
env.seed(1); torch.manual_seed(1);
discounted_reward = []
#build policy network
def data_preprocess(image_frames):
    image_frames = rgb2gray(image_frames)

    normalize_image = image_frames/255.0
    preprocessed_frame = transform.resize(normalize_image, [84,84], mode = 'reflect')



    return preprocessed_frame
stack_size = 4
stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
#returns stacked_frames-->Type: <type 'collections.deque'>  Size: (84,84,4)

#stacked_state --> Type: np.ndarray                 Size: (84,84,4)
def stack_images(stacked_frames, state, new_episode):
    frame = data_preprocess(state)
    if new_episode:
        stacked_frames = deque([torch.zeros(state.shape) for i in range(stack_size)], maxlen=4)

        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames, axis = 0)
        #print(stacked_state.shape)
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis = 0)
    return stacked_state, stacked_frames

class PolicyConvNet(nn.Module):
    """docstring for PolicyConvNet."""
    def __init__(self):
        super(PolicyConvNet, self).__init__()

        self.conv1 = nn.Conv2d(4,32,8,stride = 4)
        self.conv2 = nn.Conv2d(32,64,4,stride = 2)
        self.conv3 = nn.Conv2d(64,64,3,stride = 1)
        self.fc1 = nn.Linear(7*7*64,512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self,x):
        x = (F.relu(self.conv1(x)))

        x = (F.relu(self.conv2(x)))
        x = (F.relu(self.conv3(x)))

        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        vels = x
        x = F.softmax(self.fc2(x),dim=-1)
        # model = nn.Sequential(self.conv1, nn.ReLU(),self.conv2, nn.ReLU(),self.conv3, nn.ReLU(),self.fc1, nn.ReLU(),
        #                       self.fc2, nn.Softmax(dim=-1))
        return vels,x


#initialize policy conv net
policy_conv_net = PolicyConvNet()
optimizer_conv = optim.Adam(policy_conv_net.parameters(), lr = 0.01)

def plot_durations(total_rewards):
    plt.figure(2)
    plt.clf()
    total_rewards_plot = torch.FloatTensor(total_rewards)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.plot(total_rewards_plot.numpy())
    # Take 100 episode averages and plot them too
    # if len(total_rewards_plot) >= 100:
    #     means = total_rewards_plot.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy())

    plt.pause(0.001)
def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy_conv_net(state)
    m = Categorical(probs)
    action = m.sample()
    log_probabilities = m.log_prob(action)
    return action.item(), log_probabilities


    return action,log_probs
def discounted(r, gamma = 0.99):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  discounted_reward = []
  running_add = 0
  for t in reversed(range(0, r.size)):
    #if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
    return discounted_r

def main():
    stack_size = 4
    total_rewards = []
    use_conv = True
    num_episodes = 5000
    rewards = []
    running_reward = None
    reward_sum = 0
    log_prob_actions = []

    stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
    for i_episode in range(num_episodes):
        state = env.reset()
        print("!!!!!!!!!!!!!!!!!!!!!!")
        print(env.action_space.shape[0])


        if use_conv == True:
            state, stacked_frames = stack_images(stacked_frames, state, new_episode=True)
        else:
            # state = preprocess(state)
            state = env.reset()

        for t in range(10000):  # Don't infinite loop while learning
            action, log_actions = select_action(state)
            state, reward, done, _ = env.step(action)
            if use_conv == True:
                state, stacked_frames = stack_images(stacked_frames, state, new_episode=False)
            # else:

                # state = preprocess(state)
            rewards.append(reward)
            log_prob_actions.append(log_actions)
            reward_sum += reward
            if done:
                episode_logprobs = Variable(torch.Tensor(log_prob_actions), requires_grad=True)
                episode_rewards = (rewards)
                log_prob_actions = []
                rewards = []
                discounted_ep_reward = discounted(np.array(episode_rewards))
                d = discounted_ep_reward
                discounted_ep_reward = torch.Tensor(discounted_ep_reward)

                discounted_ep_reward = discounted_ep_reward - torch.mean(discounted_ep_reward)
                discounted_ep_reward = discounted_ep_reward/torch.std(discounted_ep_reward)
                episode_loss = -(episode_logprobs)*(discounted_ep_reward)
                episode_loss = (Variable(torch.Tensor(episode_loss), requires_grad=True)).sum()
                optimizer_conv.zero_grad()
                episode_loss.backward()
                optimizer_conv.step()
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print(f'Running Reward after episode {i_episode} is {d.sum()}')


                break






if __name__ == '__main__':
    main()
