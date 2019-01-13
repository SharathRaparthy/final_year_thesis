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


env = gym.make('CartPole-v0')
env.seed(1); torch.manual_seed(1);
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
class PolicyNet(nn.Module):
    """docstring forPolicyNet."""
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.input = nn.Linear(self.state_size,64)

        self.output = nn.Linear(64, self.action_size)
        self.reward_history = []
        self.policy_history = []
        #self.policy_history = []

    def forward(self,x):
        model = nn.Sequential(self.input, nn.ReLU(),
                          self.output, nn.Softmax(dim=-1))
        return model(x)


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

policy_net = PolicyNet()
learning_rate = 0.01
optimizer = optim.Adam(policy_net.parameters(), lr = learning_rate)
#initialize policy conv net
policy_conv_net = PolicyConvNet().cuda()
optimizer_conv = optim.Adam(policy_conv_net.parameters(), lr = learning_rate)
episode_durations = []
total_rewards = []
# def plot_durations():
#     plt.figure(2)
#     plt.clf()
#     durations_t = torch.FloatTensor(total_rewards)
#     plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Duration')
#     plt.plot(durations_t.numpy())
#     # Take 100 episode averages and plot them too
#     if len(durations_t) >= 100:
#         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
#         means = torch.cat((torch.zeros(99), means))
#         plt.plot(means.numpy())
#
#     plt.pause(0.001)
def pg_select_action(state):

    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = policy_net(Variable(state)).detach()

    c = Categorical(state)
    action = c.sample()
    # policy_net.policy_history.append(c.log_prob(action))
    logprob = c.log_prob(action)


    return action, logprob
def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i]
                  for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()

def main():
    batch_actions = []
    batch_rewards = []
    batch_states = []
    batch_logprob = []
    batch_size = 10
    number_of_episodes = 2000
    running_reward = 10
    batch_counter = 1
    # stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)


    for episodes in range(number_of_episodes):
        state = env.reset()
        reward_plots = []
        states = []
        actions = []
        log_probs = []

        policy_net.reward_history = []
        policy_net.policy_history = []
        done = False
        # stacked_state, stacked_frames = stack_images(stacked_frames, state, True)
        while done == False:

            # stacked_state, stacked_frames = stack_images(stacked_frames, state, False)

            # state = np.expand_dims(stacked_state, axis=0)

            action,logprob = pg_select_action(state)
            action = action.cpu()
            #print(action)
            next_state, reward, done, _ = env.step(action.numpy().astype(int))
            reward_plots.append(reward)
            states.append(state)
            actions.append(action)
            log_probs.append(logprob)
            policy_net.reward_history.append(reward)

            state = next_state
            if done:
                batch_rewards.extend(discount_rewards(policy_net.reward_history, gamma=0.99))
                # batch_actions.extend(actions.numpy())
                batch_states.extend(states)
                batch_logprob.extend(log_probs)
                total_rewards.append(sum(reward_plots))
                batch_counter += 1

                if batch_counter ==  batch_size:


                    reward_t = torch.FloatTensor(batch_rewards)
                    states_t = torch.FloatTensor(batch_states)
                    actions_t = torch.FloatTensor(batch_actions)
                    prob_t = torch.FloatTensor(batch_logprob)

                    optimizer.zero_grad()
                    # for i in range(len(batch_states)):
                    #     state = batch_states[i]
                    #     action = Variable(torch.FloatTensor(batch_actions[i]))
                    #     reward = batch_rewards[i]
                    #     probs = pg_select_action(state)

                    loss = -(Variable(reward_t, requires_grad  = True) * Variable(prob_t,requires_grad  = True)).mean()
                    print(loss)
                    loss.backward()
                    optimizer.step()
                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_logprob = []



                    batch_counter = 1
            print("\rEp: {} Average of last 10: {:.2f}".format(
                episodes + 1, np.mean(total_rewards[-10:])), end="")
            plt.figure(2)
            plt.clf()
            plt.plot(total_rewards)
            plt.pause(0.001)
    return total_rewards
total_rewards=main()
