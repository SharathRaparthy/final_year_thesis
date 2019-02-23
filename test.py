import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.simulator import Simulator
from gym_duckietown.wrappers import UndistortWrapper
import argparse
import numpy as np
import time
import pickle
# parser = argparse.ArgumentParser()
# parser.add_argument('--env-name', default=None)
# parser.add_argument('--map-name', default='udem1')
# parser.add_argument('--distortion', default=False, action='store_true')
# parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
# parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
# parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
# parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
# args = parser.parse_args()
#
# if args.env_name is None:
#     env = DuckietownEnv(
#         map_name = args.map_name,
#         draw_curve = args.draw_curve,
#         draw_bbox = args.draw_bbox,
#         domain_rand = args.domain_rand,
#         frame_skip = args.frame_skip,
#         distortion = args.distortion,
#     )
# else:
#     env = gym.make(args.env_name)
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)

    def action(self, action):
        action_ = [action[0] * 0.8, action[1]]
        return action_

def sigmoid(x):

    return 1/(1+np.exp(-x))
env = gym.make("Duckietown-small_loop-v0")
# env = launch_env()
env = ActionWrapper(env)
observation, tile_idx = env.reset()
# print(observation)

tile = []
tile.append(tile_idx)
for i in range(10000):
  action = env.action_space.sample() # your agent here (this takes random actions)
  action[0] = sigmoid(action[0])
  # env.render()
  observation, reward, done, info = env.step(action)
  if done:
      _, tile_idx = env.reset()
      tile.append(tile_idx)
import matplotlib.pyplot as plt
plt.hist(tile)
plt.show()
