import gym_duckietown
import gym
import torch
import gym
from gym import spaces
import numpy as np
import time
from ddpg_notebook import Actor

class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(120, 160, 3)):
        super(ResizeWrapper, self).__init__(env)
        self.observation_space.shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            shape,
            dtype=self.observation_space.dtype)
        self.shape = shape

    def observation(self, observation):
        from scipy.misc import imresize
        return imresize(observation, self.shape)


class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)


class ImgWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ImgWrapper, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -10
        elif reward > 0:
            reward += 10
        else:
            reward += 4

        return reward


# this is needed because at max speed the duckie can't turn anymore
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)

    def action(self, action):
        action_ = [action[0] * 0.8, action[0]]
        return action_


# In[ ]:


env = gym.make('Duckietown-udem1-v0')

env = ResizeWrapper(env)
env = NormalizeWrapper(env)
env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
env = ActionWrapper(env)
env = DtRewardWrapper(env)
state_size = env.observation_space.shape
action_size = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

actor_agent = Actor(state_size, action_size,max_action)
actor_path = torch.load('/home/ivlabs/users/sharath/final_year_thesis/checkpoint_actor_EXP2.pth')
actor_agent.load_state_dict(actor_path)
state = env.reset()
with torch.no_grad():
    while True:
        state = env.reset()
        rewards = []
        while True:
            env.render()
            state = torch.from_numpy(state).float()
            state = state.unsqueeze(0)
            action = actor_agent(state).cpu().data.numpy()
            # action[0][0] = np.clip(action[0][0],0,1)

            state, reward, done, _ = env.step(action[0])
            rewards.append(reward)
            if done:
                break
        print("Mean Episode Reward:", np.mean(rewards))
