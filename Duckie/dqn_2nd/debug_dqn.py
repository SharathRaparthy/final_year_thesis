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

##import libraries related to DuckietownEnv
import argparse
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.simulator import Simulator
#from gym_duckietown.wrappers import UndistortWrapper

# if gpu is to be used
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

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

#actions

up = np.array([0.44, 0.0])
stop = np.array([0.0,0.0])
left = np.array([0.35,-1])
right = np.array([0.35, +1])

possible_actions = [up,stop,left,right]
#possible_actions = np.array(possible_actions) #<type 'numpy.ndarray'>  (4,2)

mapped_actions = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])


#preprocess the data
#returns (84,84,1) -> HWC
def data_preprocess(image_frames):
    gray_image = rgb2gray(image_frames)
    normalize_image = gray_image/255.0
    preprocessed_frame = transform.resize(normalize_image, [84,84])
    return preprocessed_frame

###Create stacks of 4 frames.
stack_size = 4
stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
#returns stacked_frames-->Type: <type 'collections.deque'>  Size: (84,84,4)

#        stacked_state --> Type: np.ndarray                 Size: (84,84,4)
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
total_episodes = 2000
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
    def __init__(self, state_size, action_size = 4):
        super(DeepQNet, self).__init__()
        self.conv1 = nn.Conv2d(4,32,8,stride = 4)
        self.conv2 = nn.Conv2d(32,64,4,stride = 2)
        self.conv3 = nn.Conv2d(64,64,3,stride = 1)
        self.fc1 = nn.Linear(7*7*64,512)
        self.fc2 = nn.Linear(512, action_size)
        self.dropout = nn.Dropout(0.2)
    #state is passed as x (batch_size,4,84,84)
    #returns Tensor.i.e. q_values  (64,4,1) -->CHW   #doubtful
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0),-1) #flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    #Predict action depending upon exploitation-Exploration tradeoff
    def act(self,state, epsilon, e):
        #exploit
        if e > epsilon:
            #state = Variable(torch.FloatTensor(state).cuda())
            state = Variable(torch.FloatTensor(state))
            state = state.unsqueeze(0)
            #state size should be (None,4,84,84) i.e. 4D inputs
            #print("state for exploitation is:",state.shape)
            q_values = self.forward(state)
            #print(q_values.shape) #(64,4,1)
            action = possible_actions[q_values.max(1)[1]] #check for vector column or row
            #print("action shape at exploitation:",action.shape)#(64,1) doubtful
        else:
            action = possible_actions[random.randrange(len(possible_actions))]
            print("Exploration going on")
        return action

####Calculate Loss for training
def compute_loss(batch_size, state_batch, action_batch, reward_batch, next_state_batch, done_episode):
    #convert state,action,reward,next_state,done_episode to tensor Variable
    '''
    state_batch = Variable(torch.FloatTensor(state_batch).cuda())
    action_batch = Variable(torch.LongTensor(action_batch).cuda())
    reward_batch = Variable(torch.FloatTensor(reward_batch).cuda())
    next_state_batch = Variable(torch.FloatTensor(next_state_batch).cuda())
    done_episode = Variable(torch.FloatTensor(done_episode).cuda())
    '''
    #print("going into loss")
    state_batch = Variable(torch.FloatTensor(state_batch))
    action_batch = Variable(torch.FloatTensor(action_batch))
    reward_batch = Variable(torch.FloatTensor(reward_batch))
    next_state_batch = Variable(torch.FloatTensor(next_state_batch))
    done_episode = Variable(torch.FloatTensor(done_episode))

    #print("action_batch size:",action_batch.shape)
    #print("DQN(state_batch)",DQN(state_batch).shape)
    #action = action.unsqueeze(1)
    #q_values= DQN(state_batch).gather(1, action_batch)#q_values --> (64,1) after gathering
    #print("DQN(state_batch)",DQN(state_batch).shape)
    #print("q_values:", q_values.shape)
    #print("action_batch",action_batch.shape)

    q_values = torch.bmm(action_batch.unsqueeze(1),DQN(state_batch).unsqueeze(2))

    next_q_values = DQN(next_state_batch) #nex_q_values --> (64,4)

    #Calculate target q_value using Bellman Function
    target_q_values = reward_batch + gamma*next_q_values.max(1)[0].detach()

    #Loss Calculate
    loss = F.smooth_l1_loss(q_values, target_q_values) #unsqueeze not done
    #loss = ( Variable(expected_q_value) - q_value).pow(2).mean()
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
        #print("action_batch_size_from smaple:",action.Size)
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)


replay_buffer = BufferReplay(memory_size)
#initialise network and optimizers
state = env.reset()
#DQN = DeepQNet(state).cuda()
DQN = DeepQNet(state)
DQN_optimizer = optim.Adam(DQN.parameters(), lr = 0.0002)
loss_list = []


#####------------training loop-----------------#########
for i in range(total_episodes):
    step = 0
    decay_step = 0
    episode_rewards = []
    state = env.reset() #(84,84,3)
    state = state.transpose((2, 0, 1))#convert from HWC to CHW
    stacked_state, stacked_frames = stack_images(stacked_frames, state, True) #(4,84,84)

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
            #state = stacked_state #(4,84,84)

            action = DQN.act(stacked_state, epsilon, e)

            print("Action:", action)
            next_state, reward, done, _ = env.step(action)
            print("Reward:", reward)

            next_stacked_state, stacked_frames = stack_images(stacked_frames, next_state.transpose((2, 0, 1)), False)

            #mapped duckietown action to mapped action
            if np.array_equal(action,up):
                action = mapped_actions[0]
            elif np.array_equal(action,stop):
                action = mapped_actions[1]
            elif np.array_equal(action,left):
                action = mapped_actions[2]
            elif np.array_equal(action,right):
                action = mapped_actions[3]


            #push Experience to buffer
            replay_buffer.push(stacked_state, action, reward, next_stacked_state, done)
            episode_rewards.append(reward)

            #For LOSS ONLY
            #sample mini_batch from buffer:   state_batch and next_state_batch --> (64,4,84,84)  action_batch and reward_batch --> (64,1)
            state_batch, action_batch, reward_batch, next_state_batch, done = replay_buffer.sample(batch_size)
            #print("action_batch:",action_batch)
            #pass mini-batches for Calculation of loss
            DQN_optimizer.zero_grad()
            loss = compute_loss(batch_size, state_batch, action_batch, reward_batch, next_state_batch, done)
            loss.backward()
            DQN_optimizer.step()

            loss_list.append(loss)
            print('Episode number is {0}/{1} | Step number is {2} | Loss is {3}'.format(i,total_episodes,step,loss))
            #print("Loss",loss)
        else:
            #for making agent explore
            e = (random.random())%epsilon
            action = DQN.act(stacked_state, epsilon, e)
            print(action)
            next_state, reward, done, _ = env.step(action)

            next_stacked_state, stacked_frames = stack_images(stacked_frames, next_state.transpose((2, 0, 1)), False)

            #mapped duckietown action to mapped action
            if np.array_equal(action,up):
                action = mapped_actions[0]
            elif np.array_equal(action,stop):
                action = mapped_actions[1]
            elif np.array_equal(action,left):
                action = mapped_actions[2]
            elif np.array_equal(action,right):
                action = mapped_actions[3]

            replay_buffer.push(stacked_state, action, reward, next_stacked_state, done)
            episode_rewards.append(reward)

        stacked_state = next_stacked_state
        if reward<0.05:
            break



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

#(84,84,3)
state = env.reset()

stacked_state, stacked_frames = stack_images(stacked_frames, state.transpose((2, 0, 1)), True)
for i in range(eval_steps):
    e = 0.8
    epsilon = 0.5
    action = DQN.act(stacked_state, epsilon,e)
    next_state, reward, done, _ = env.step(action)
    next_stacked_state, stacked_frames = stack_images(stacked_frames, next_state.transpose((2, 0, 1)), False)
    print('Eval_steps number is {0}'.format(eval_steps))
    eval_rewards.append(reward)
    stacked_state = next_stacked_state
    env.rener()

print('Evaluation finished')
