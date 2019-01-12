import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
import gym
from torch.autograd import Variable
env = gym.make('CartPole-v0')
env.seed(1); torch.manual_seed(1);
#build policy network

class PolicyNet(nn.Module):
    """docstring forPolicyNet."""
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.input = nn.Linear(self.state_size,128)
        self.output = nn.Linear(128, self.action_size)
        self.reward_history = []
        self.policy_history = []
        #self.policy_history = []

    def forward(self,x):
        model = nn.Sequential(self.input, nn.ReLU(),
                          self.output, nn.Softmax(dim=-1))
        return model(x)
policy_net = PolicyNet().cuda()
print(policy_net)
learning_rate = 0.01
optimizer = optim.Adam(policy_net.parameters(), lr = learning_rate)

def pg_select_action(state):
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = policy_net(Variable(state.cuda()))
    c = Categorical(state)
    action = c.sample()
    policy_net.policy_history.append(c.log_prob(action))

    return action
def main():
    number_of_episodes = 10000
    running_reward = 10

    for episodes in range(number_of_episodes):
        state = env.reset()
        for time in range(1000):



            action = pg_select_action(state)
            action = action.cpu()
            #print(action)
            next_state, reward, done, _ = env.step(action.numpy().astype(int))
            policy_net.reward_history.append(reward)
            state = next_state
            if done:
                print(done)
                break
    #calculate the loss and update the weights
    #calculate the discounted rewards
        R = 0
        gamma = 0.99
        reward = []
        for rewards in policy_net.reward_history[::-1]:
            R = rewards + gamma*R
            reward.insert(0,R)

        reward = torch.FloatTensor(reward)
        means = torch.mean(reward)
        std = torch.std(reward)
        reward = (reward - means)/std

        policies = torch.FloatTensor(policy_net.policy_history)
        optimizer.zero_grad()
        loss = (torch.sum(torch.mul(policies,Variable(reward, requires_grad=True)).mul(-1),-1))
        loss.backward()
        optimizer.step()
        running_reward = (running_reward * 0.99) + (time * 0.01)
        if episodes % 50 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(episodes, time, running_reward))
main()
