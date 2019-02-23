import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.simulator import Simulator
from gym_duckietown.wrappers import UndistortWrapper
import argparse
import numpy as np
import time
#
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

def launch_env(id=None):
    env = None
    if id is None:
        from gym_duckietown.simulator import Simulator
        env = Simulator(
            seed=123, # random seed
            map_name="small_loop",
            max_steps=500001, # we don't want the gym to reset itself
            domain_rand=0,
            camera_width=640,
            camera_height=480,
            accept_start_angle_deg=2, # start close to straight
            full_transparency=True,
            distortion=False,
        )
    else:
        env = gym.make(id)

    return env
def sigmoid(x):

    return 1/(1+np.exp(-x))
env = gym.make("Duckietown-small_loop-v0")
env = launch_env()
env = ActionWrapper(env)
observation = env.reset()
# print(observation)
for i in range(1000):
  action = env.action_space.sample() # your agent here (this takes random actions)

  env.render()
  action[0] = sigmoid(action[0])
  # action = np.array([0.95,-1])
  observation, reward, done, info = env.step(action)
  if done:
      env.reset()
  time.sleep(0.1)
  print(action)
