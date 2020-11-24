import gym
from gym import spaces

class WarehouseEnv(gym.Env):
    def __init__(self, map):
        super().__init__()

        self.map = map

    def step(self, action):
        observation = reward = done = info = None

    def reset(self):
        observation = None
        return observation

    def render(self, mode='human'):
        pass

    def close(self):
        pass