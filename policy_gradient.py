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




env = gym.make('Pong-v0')
env.seed(1); torch.manual_seed(1);
#build policy network
def data_preprocess(image_frames):

    normalize_image = image_frames/255.0
    preprocessed_frame = transform.resize(normalize_image, [84,84], mode = 'reflect')
    preprocessed_frame = np.transpose(preprocessed_frame, (2, 1, 0))
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis = 0)


    return preprocessed_frame
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
        self.conv1 = nn.Conv2d(3,32,8,stride = 4)
        self.conv2 = nn.Conv2d(32,64,4,stride = 2)
        self.conv3 = nn.Conv2d(64,64,3,stride = 1)
        self.fc1 = nn.Linear(7*7*64,512)
        self.fc2 = nn.Linear(512, self.action_size)
        self.dropout = nn.Dropout(0.2)
        self.reward_history = []
        self.policy_history = []
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
policy_conv_net = PolicyConvNet()
optimizer_conv = optim.Adam(policy_conv_net.parameters(), lr = learning_rate)

def pg_select_action(state):
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = policy_conv_net(Variable(state))
    c = Categorical(state)
    action = c.sample()
    policy_conv_net.policy_history.append(c.log_prob(action))

    return action
def main():
    number_of_episodes = 10000
    running_reward = 10

    for episodes in range(number_of_episodes):
        state = env.reset()

        for time in range(1000):
            state = data_preprocess(state)


            action = pg_select_action(state)
            #print(action)
            next_state, reward, done, _ = env.step(action.numpy().astype(int))
            policy_conv_net.reward_history.append(reward)
            state = next_state
            if done:
                break
    #calculate the loss and update the weights
    #calculate the discounted rewards
        R = 0
        gamma = 0.99
        reward = []
        for rewards in policy_conv_net.reward_history[::-1]:
            R = rewards + gamma*R
            reward.insert(0,R)

        reward = torch.FloatTensor(reward)
        means = torch.mean(reward)
        std = torch.std(reward)
        reward = (reward - means)/std

        policies = torch.FloatTensor(policy_conv_net.policy_history)
        optimizer.zero_grad()
        loss = (torch.sum(torch.mul(policies,Variable(reward, requires_grad=True)).mul(-1),-1))
        loss.backward()
        optimizer.step()
        running_reward = (running_reward * 0.99) + (time * 0.01)
        if episodes % 50 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(episodes, time, running_reward))
main()
