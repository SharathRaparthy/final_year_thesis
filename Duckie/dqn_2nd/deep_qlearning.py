.aimport numpy as np
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

##import libraries related to DuckietownEnv
import argparse
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.simulator import Simulator
from gym_duckietown.wrappers import UndistortWrapper

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####All arguments
parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
#parser.add_argument('--map-name', default='udem1')
#parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
#parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
#parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=3, type=int, help='number of frames to skip')
args = parser.parse_args()

if args.env_name is None:
    env = DuckietownEnv(
        frame_skip=args.frame_skip
    )
else:
    env = gym.make(args.env_name)
#initialise the environment

#env = gym.make('Assault-v0')
#one_hot encoding of actions
#possible_actions = np.array(np.identity(env.action_space.n, dtype=int)).tolist()
up = np.array([0.44, 0.0])
stop = np.array([0.0,0.0])
left = np.array([0.35,-1])
right = np.array([0.35, +1])
possible_actions = [up,stop,left,right]

#preprocess the data
def data_preprocess(image_frames):
    gray_image = rgb2gray(image_frames)
    normalize_image = gray_image/255.0
    preprocessed_frame = transform.resize(normalize_image, [84,84])
    return preprocessed_frame

###Create stacks of 4 frames
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

####Hyper-parameters
#Training parameters
action_size = 4 #[left,rigth,up,stop]
gamma = 0.95
total_episodes = 20
total_steps = 15000
batch_size = 64
#Exploration-Exploitation parameters
explore_start = 0.9
explore_stop = 0.01
decay_rate = 0.00001

#Experience Replay parameter
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
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0),-1) #flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def act(self,state, epsilon, e):
        if e > epsilon:
            state = Variable(torch.FloatTensor(state).cuda())
            print('exploitation')
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0]
        else:
            action = possible_actions[random.randrange(len(possible_actions))]
            print(action)
        return action

####Calculate Loss for training
def compute_loss(batch_size, state, action, reward, next_state, done_episode):
    #convert state,action,reward,next_state,done_episode to tensor Variable
    state = Variable(torch.FloatTensor(state).cuda())
    action = Variable(torch.LongTensor(action).cuda())
    reward = Variable(torch.FloatTensor(reward).cuda())
    next_state = Variable(torch.FloatTensor(next_state).cuda())
    done_episode = Variable(torch.FloatTensor(done_episode).cuda())


    #action = action.unsqueeze(1)
    q_values = DQN(state)
    q_value = q_values.max(1)[0]

    if done_episode[0]:
    	next_q_values = reward
    else:
        next_q_values = DQN(next_state)



    #Bellman Function
    expected_q_value = reward + gamma*next_q_values.max(1)[0].detach()
    #expected_q_value = expected_q_value.unsqueeze(1)

    #Loss Calculate
    #loss = F.smooth_l1_loss(q_value, expected_q_value)
    loss = ( Variable(expected_q_value) - q_value).pow(2).mean()



    return loss

class BufferReplay(object):
    """docstring forBufferReplay."""
    def __init__(self, buffersize):
        super(BufferReplay, self).__init__()
        self.buffer = deque(maxlen=memory_size)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)


replay_buffer = BufferReplay(memory_size)
#initialise network and optimizers
state = env.reset()
DQN = DeepQNet(state).cuda()
DQN_optimizer = optim.Adam(DQN.parameters(), lr = 0.0002)
loss_list = []


#####------------training loop-----------------#########
for i in range(total_episodes):
    step = 0
    decay_step = 0
    episode_rewards = []
    state = env.reset()
    state = state.transpose((2, 0, 1))#convert from HWC to CHW
    state, stacked_frames = stack_images(stacked_frames, state, True)

    while step < total_steps:
        #epsilon greedy approach
        step = step + 1
        decay_step += 1
        #exponential decay of exploration rate
        epsilon = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

        #if buffer size crosses batch_size
        if len(replay_buffer) > batch_size:
            if len(replay_buffer) == batch_size:
                loss = 0
            e = random.random()
            state_small = state
            #sample mini_batch from buffer
            state, action_sample, reward_sample, next_state_sample, done_sample = replay_buffer.sample(batch_size)
            #print("argato",state.shape)
            action = DQN.act(state, epsilon, e)
            print("Action:", action)
            next_state, reward, done, _ = env.step(action)
            print("Reward:", reward)
            next_state, stacked_frames = stack_images(stacked_frames, next_state.transpose((2, 0, 1)), False)
            #push Experience to buffer
            replay_buffer.push(state_small, action, reward, next_state, done)
            episode_rewards.append(reward)

            DQN_optimizer.zero_grad()
            loss = compute_loss(batch_size, state, action_sample, reward_sample, next_state_sample, done_sample)
            loss.backward()
            DQN_optimizer.step()

            loss_list.append(loss)
            print(f'Episode number is {i}/{total_episodes} | Step number is {step} | Loss is {loss}')
        else:
            #for making agent explore
            e = (random.random())%epsilon
            action = DQN.act(state, epsilon, e)
            print(action)
            next_state, reward, done, _ = env.step(action)
            next_state, stacked_frames = stack_images(stacked_frames, next_state.transpose((2, 0, 1)), False)
            replay_buffer.push(state, action, reward, next_state, done)
            episode_rewards.append(reward)
        if reward<0.05:
            break
        state = next_state


#-------------------------------------------------evaluation------------------------------------------#

####Save trained_parameters
PATH =  "./trained_parameters"
torch.save(DQN.state_dict(),PATH)

#Evaluation parameters
eval_steps = 50000
eval_rewards = []
####Load trained_parameters and test trained agent
DQN.load_state_dict(torch.load(PATH))
DQN.eval()

state = env.reset()
state, stacked_frames = stack_images(stacked_frames, state.transpose((2, 0, 1)), True)
for i in range(eval_steps):
    e = 0.8
    epsilon = 0.5
    action = DQN.act(state, epsilon,e)
    next_state, reward, done, _ = env.step(action)
    next_state, stacked_frames = stack_images(stacked_frames, next_state.transpose((2, 0, 1)), False)
    env.render()
    eval_rewards.append(reward)
    state = next_state

print('Evaluation finished')
