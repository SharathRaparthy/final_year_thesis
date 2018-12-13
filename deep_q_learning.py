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
#initialise the environment

env = gym.make('CartPole-v0')
#one_hot encoding of actions
possible_actions = np.array(np.identity(env.action_space.n, dtype=int)).tolist()

#preprocess the data
def data_preprocess(image_frames):
    gray_image = rgb2gray(image_frames)
    normalize_image = gray_image/255.0
    preprocessed_frame = transform.resize(normalized_frame, [84,84])


    return preprocessed_frame


stack_size = 4


def stack_images(stack_size, image_frame, new_episode):

    frame = data_preprocess(image_frame)



    if new_episode:
        stacked_images = deque([torch.zeros(image_frames.shape) for i in range(stack_size)], maxlen=4)

        stacked_images.append(frame, axis=1)
        stacked_images.append(frame, axis=1)
        stacked_images.append(frame, axis=1)
        stacked_images.append(frame, axis=1)
    else:
        stacked_images.append(frame, axis=1)
    return stacked_images

#Hyper-parameters
action_size = env.action_space.n
learning_rate = 0.00025
total_episodes = 50
total_steps = 50000
batch_size = 64
explore_start = 1.0
explore_stop = 0.01
deacay_rate = 0.00001

discounted_rate = 0.9

memory_size = 1000000
def convolution_output(width, height, stride, padding, kernel):
    width = ((width - kernel + 2*padding)/stride) + 1
    height = ((height - kernel + 2*padding)/stride) + 1
    return width, height


#build network architecture
class DeepQNet(nn.Module):
    """docstring forDeepQNet."""
    def __init__(self, state_size, action_size = 3):
        super(DeepQNet, self).__init__()
        self.conv1 = nn.Conv2d(4,16,8,stride = 4)
        self.conv2 = nn.Conv2d(16,32,4,stride = 2)
        self.fc1 = nn.Linear(256,100)
        self.fc2 = nn.Linear(100, action_size)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.2)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1,x.shape[2]*x.shape[3])
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def act(self,state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state), volatile=True)
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(env.action_space.n)
        return action
def compute_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    state = Variable(torch.FloatTensor(state))
    action = Variable(torch.FloatTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    next_state = Variable(torch.FloatTensor(next_state))
    done = Variable(torch.FloatTensor(done))

    q_values = DeepQNet(state)
    next_q_values = DeepQNet(next_state)

    expected_q_value = reward + gamma*next_q_values.max(1)[0]
    loss = (q_value - Variable(expected_q_value)).pow(2).mean()

    return loss

class BufferReplay(object):
    """docstring forBufferReplay."""
    def __init__(self, buffersize):
        super(BufferReplay, self).__init__()
        self.buffer = deque(maxlen=memory_size)
    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        buffer.append(state, action, reward, next_state, done)
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))

        return state, action, reward, next_state, done
    def __len__(self):
        return len(buffer)
replay_buffer = BufferReplay(memory_size)
#initialise network and optimizers
state = env.reset()
DQN = DeepQNet(state)
DQN_optimizer = optim.Adam(DQN.parameters(), lr = learning_rate)
loss = []
#training loop
buffer_loop = 500

for i in range(total_episodes):
    step = 0
    episode_rewards = []
    state = env.reset()
    state = stack_images(4, state, True)

    while step < total_steps:
        step = step + 1
        decay_step += 1
        epsilon = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        action = DQN.act(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        episode_rewards.append(reward)
        if done:
            total_steps = steps

        if len(replay_buffer) > batch_size:
            DQN_optimizer.zero_grad()
            loss = compute_loss(batch_size)
            loss.backward()
            DQN_optimizer.step()
            loss.append(loss)
