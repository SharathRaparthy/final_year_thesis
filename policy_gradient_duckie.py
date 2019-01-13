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

policy_net = PolicyNet()
learning_rate = 0.01
optimizer = optim.Adam(policy_net.parameters(), lr = learning_rate)
#initialize policy conv net
policy_conv_net = PolicyConvNet().cuda()
optimizer_conv = optim.Adam(policy_conv_net.parameters(), lr = learning_rate)

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
def pg_select_action(state):

    state = torch.from_numpy(state).type(torch.FloatTensor)
    vels, probs = policy_conv_net(Variable(state.cuda()))



    c = Categorical(probs)
    # action = c.sample()
    print(action)
    log_probs = c.log_prob(action)

    # policy_conv_net.policy_history.append(torch.log(state))


    return action,log_probs
def main():
    number_of_episodes = 10000
    running_reward = 10
    stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)


    for episodes in range(number_of_episodes):

        state = env.reset()
        stacked_state, stacked_frames = stack_images(stacked_frames, state, True)

        rewards = []
        total_rewards = []
        log_probabilities = []

        for time in count():
            env.render()


            state = np.expand_dims(stacked_state, axis=0)

            action,log_probs = pg_select_action(state)
            action = action.cpu()
            print(action)

            #print(action)
            next_state, reward, done, _ = env.step(action.detach().numpy())

            rewards.append(reward)
            log_probabilities.append(log_probs)
            state = next_state
            stacked_state, stacked_frames = stack_images(stacked_frames, state, False)
            if done:
                break



    #calculate the loss and update the weights
    #calculate the discounted rewards
        R = 0
        gamma = 0.99
        reward = []
        for rewards in rewards[::-1]:
            R = rewards + gamma*R
            reward.insert(0,R)
        discounted_reward.append(sum(reward))
        plot_durations(discounted_reward)

        reward = torch.FloatTensor(reward)
        reward = Variable(reward, requires_grad=True)
        log_probabilities = torch.FloatTensor(log_probabilities)
        means = torch.mean(reward)
        std = torch.std(reward)
        reward = (reward - means)/std

        selected_logprobs = reward*log_probabilities
        optimizer_conv.zero_grad()
        loss = -selected_logprobs.mean()
        loss.backward()
        optimizer_conv.step()

        running_reward = (running_reward * 0.99) + (time * 0.01)

        if episodes % 50 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(episodes, time, running_reward))
main()
