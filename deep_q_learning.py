import numpy as np
from collections import deque
from skimage.color import rgb2gray
import torch
import gym
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import random
from skimage import transform
import torch.nn.functional as F


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#initialise the environment

env = gym.make('Assault-v0')
#one_hot encoding of actions
possible_actions = np.array(np.identity(env.action_space.n, dtype=int)).tolist()

#preprocess the data
def data_preprocess(image_frames):
    gray_image = rgb2gray(image_frames)
    normalize_image = gray_image/255.0
    preprocessed_frame = transform.resize(normalize_image, [84,84])


    return preprocessed_frame


stack_size = 4

stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
def stack_images(stacked_frames, state, new_episode):

    frame = data_preprocess(state)
    stack_size = 4



    if new_episode:
        stacked_frames = deque([torch.zeros(state.shape) for i in range(stack_size)], maxlen=4)

        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames, axis = 0)
        print(stacked_state.shape)
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis = 0)
    return stacked_state, stacked_frames

#Hyper-parameters
action_size = env.action_space.n
gamma = 0.95
total_episodes = 50
total_steps = 50000
batch_size = 64
explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.00001

discounted_rate = 0.9

memory_size = 1000000


#build network architecture
class DeepQNet(nn.Module):
    """docstring forDeepQNet."""
    def __init__(self, state_size, action_size = action_size):
        super(DeepQNet, self).__init__()
        self.conv1 = nn.Conv2d(4,32,8,stride = 4)
        self.conv2 = nn.Conv2d(32,64,4,stride = 2)
        self.conv3 = nn.Conv2d(64,64,3,stride = 1)
        self.fc1 = nn.Linear(7*7*64,512)
        self.fc2 = nn.Linear(512, action_size)
        self.dropout = nn.Dropout(0.2)
    def forward(self,x):
        x = (F.relu(self.conv1(x)))

        x = (F.relu(self.conv2(x)))
        x = (F.relu(self.conv3(x)))

        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def act(self,state, epsilon, e):
        if e > epsilon:
            #state, action, reward, next_state, done = batch_sampling(batch_size, done_episode)
            state = Variable(torch.FloatTensor(state), device = device, volatile=True)
            print('exploitation')
            q_value = self.forward(state)
            #action = q_value.max(1)[1].view(1, 1)
            action = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(env.action_space.n)
            print(action)
        return action



def compute_loss(batch_size, state, action, reward, next_state, done_episode):
    #state, action, reward, next_state, done = batch_sampling(batch_size, done_episode)

    #state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    state = Variable(torch.FloatTensor(state), device = device)
    action = Variable(torch.LongTensor(action), device = device)
    reward = Variable(torch.FloatTensor(reward), device = device)
    next_state = Variable(torch.FloatTensor(next_state), device = device)
    done_episode = Variable(torch.FloatTensor(done_episode), device = device)

    action = action.unsqueeze(1)
    q_values = DQN(state).gather(1, action)



    # print(action.shape)
    # action = torch.unsqueeze(action,1)
    # print(action.shape)
    # q_values = q_values.gather(1, action)

    next_q_values = DQN(next_state)

    expected_q_value = reward + gamma*next_q_values.max(1)[0].detach()
    expected_q_value = expected_q_value.unsqueeze(1)

    loss = F.smooth_l1_loss(q_values, expected_q_value)
    if done_episode[0]:
        expected_q_value = reward

    return loss

class BufferReplay(object):
    """docstring forBufferReplay."""
    def __init__(self, buffersize):
        super(BufferReplay, self).__init__()
        self.buffer = deque(maxlen=memory_size)
    def push(self, state, action, reward, next_state, done):
        #state = np.expand_dims(state, 0)
        #next_state = np.expand_dims(next_state, 0)
        self.buffer.append([state, action, reward, next_state, done])
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))

        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)


replay_buffer = BufferReplay(memory_size)
#initialise network and optimizers
state = env.reset()
DQN = DeepQNet(state).to(device)
DQN_optimizer = optim.Adam(DQN.parameters(), lr = 0.0002)
loss_list = []
#training loop
buffer_loop = 500



for i in range(total_episodes):
    step = 0
    decay_step = 0
    episode_rewards = []
    #epsilon = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    state = env.reset()
    state = state.transpose((2, 0, 1))
    state, stacked_frames = stack_images(stacked_frames, state, True)
    #action = DQN.act(state, epsilon)
    #next_state, reward, done, _ = env.step(action)
    #next_state, stacked_frames = stack_images(stacked_frames, next_state.transpose((2, 0, 1)), False)
    #state = next_state

    while step < total_steps:
        step = step + 1
        decay_step += 1
        epsilon = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)






        if len(replay_buffer) > batch_size:
            if len(replay_buffer) == batch_size:
                loss = 0
            e = random.random()
            state_small = state
            state, action_sample, reward_sample, next_state_sample, done_sample = replay_buffer.sample(batch_size)
            action = DQN.act(state, epsilon, e)
            print(action)
            next_state, reward, done, _ = env.step(action)
            print(reward)
            next_state, stacked_frames = stack_images(stacked_frames, next_state.transpose((2, 0, 1)), False)

            replay_buffer.push(state_small, action, reward, next_state, done)
            episode_rewards.append(reward)
            DQN_optimizer.zero_grad()
            loss = compute_loss(batch_size, state, action_sample, reward_sample, next_state_sample, done_sample)
            loss.backward()
            DQN_optimizer.step()
            loss_list.append(loss)
            print(f'Episode number is {i}/{total_episodes} | Step number is {step} | Loss is {loss}')
        else:
            e = (random.random())%epsilon#not correct
            action = DQN.act(state, epsilon, e)
            print(action)
            next_state, reward, done, _ = env.step(action)
            next_state, stacked_frames = stack_images(stacked_frames, next_state.transpose((2, 0, 1)), False)
            replay_buffer.push(state, action, reward, next_state, done)
            episode_rewards.append(reward)

        state = next_state
