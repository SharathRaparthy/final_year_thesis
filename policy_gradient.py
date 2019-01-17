import argparse
import gym
import numpy as np
from itertools import count
from collections import deque
from skimage.color import rgb2gray
from skimage import transform
import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('Enduro-v0')
def preprocess(image):
    image = image[35:195]
    image = image[::2,::2,0]
    image[image==144] = 0
    image[image==109] = 0
    image[image != 0] = 1

    return image.astype(np.float).ravel()

def data_preprocess(image_frames):
    image_frames = rgb2gray(image_frames)

    normalize_image = image_frames/255.0
    preprocessed_frame = transform.resize(normalize_image, [84,84], mode = 'reflect')



    return preprocessed_frame

#returns stacked_frames-->Type: <type 'collections.deque'>  Size: (84,84,4)

#stacked_state --> Type: np.ndarray                 Size: (84,84,4)
stack_size = 4
stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)

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
        self.action_size = env.action_space.n

        self.conv1 = nn.Conv2d(4,32,8,stride = 4)
        self.conv2 = nn.Conv2d(32,64,4,stride = 2)
        self.conv3 = nn.Conv2d(64,64,3,stride = 1)
        self.fc1 = nn.Linear(7*7*64,512)
        self.fc2 = nn.Linear(512, self.action_size)
        self.dropout = nn.Dropout(0.2)
        self.saved_log_probs_conv = []
        self.rewards_conv = []

    def forward(self,x):
        x = (F.relu(self.conv1(x)))

        x = (F.relu(self.conv2(x)))
        x = (F.relu(self.conv3(x)))

        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x),dim=-1)
        # model = nn.Sequential(self.conv1, nn.ReLU(),self.conv2, nn.ReLU(),self.conv3, nn.ReLU(),self.fc1, nn.ReLU(),
        #                       self.fc2, nn.Softmax(dim=-1))
        return x
policy_conv = PolicyConvNet()
optimizer_conv = optim.Adam(policy_conv.parameters(),lr=1e-2)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.action_space = env.action_space.n
        self.input = nn.Linear(6400, 200)
        self.output = nn.Linear(200, self.action_space)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.input(x))
        action_scores = self.output(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy_conv(state)
    m = Categorical(probs)
    action = m.sample()
    log_probabilities = m.log_prob(action)
    return action.item(), log_probabilities


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

def discounted(r, gamma = 0.99):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  discounted_reward = []
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
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


        if use_conv == True:
            state, stacked_frames = stack_images(stacked_frames, state, new_episode=True)
        else:
            state = preprocess(state)

        for t in range(10000):  # Don't infinite loop while learning
            action, log_actions = select_action(state)
            state, reward, done, _ = env.step(action)
            if use_conv == True:
                state, stacked_frames = stack_images(stacked_frames, state, new_episode=False)
            else:
                state = preprocess(state)
            rewards.append(reward)
            log_prob_actions.append(log_actions)
            reward_sum += reward
            if done:
                episode_logprobs = Variable(torch.Tensor(log_prob_actions), requires_grad=True)
                episode_rewards = (rewards)
                log_prob_actions = []
                rewards = []
                discounted_ep_reward = discounted(np.array(episode_rewards))
                discounted_ep_reward = torch.Tensor(discounted_ep_reward)

                discounted_ep_reward = discounted_ep_reward - torch.mean(discounted_ep_reward)
                discounted_ep_reward = discounted_ep_reward/torch.std(discounted_ep_reward)
                episode_loss = -(episode_logprobs)*(discounted_ep_reward)
                episode_loss = (Variable(torch.Tensor(episode_loss))).sum()
                optimizer_conv.zero_grad()
                episode_loss.backward()
                optimizer_conv.step()
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print(f'Running Reward after episode {i_episode} is {running_reward}')


                break






if __name__ == '__main__':
    main()
