import gym
from gym import spaces
import numpy as np
from enum import Enum


class Action(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3
    NOOP = 4


class WarehouseEnv(gym.Env):
    def __init__(self, obstacle_map, agent_map):
        super().__init__()
        assert obstacle_map.size == agent_map.size

        self.agent_state = {}
        self.agent_goal = {}
        rows, cols = np.nonzero(agent_map)
        for i, (row, col) in enumerate(zip(rows, cols)):
            self.agent_state[i] = (row, col)
            self.agent_goal[i] = None

        self.n_agents = len(self.agent_state)
        self.obstacle_map = obstacle_map
        self.agent_map = agent_map
        self.goal_map = np.zeros_like(agent_map)
        self.action_space = spaces.Discrete(5)

    def _observe(self, agent):
        goal = self.agent_goal[agent]
        state = self.agent_state[agent]

        agent_channel = np.zeros_like(self.agent_map)
        agent_channel[state] = 1
        other_agent_channel = np.zeros_like(self.agent_map)
        for k, v in self.agent_state.items():
            if k != agent:
                other_agent_channel[v] = 1
        goal_channel = np.zeros_like(self.goal_map)
        goal_channel[goal] = 1
        other_goal_channel = np.zeros_like(self.goal_map)
        for k, v in self.agent_goal.items():
            if k != agent:
                other_goal_channel[v] = 1
        return np.stack(
            [agent_channel, other_agent_channel, goal_channel, other_goal_channel],
            axis=0,
        )

    def _occupied(self, row, col):
        if self.obstacle_map[row, col] == 1:
            return True
        for _, v in self.agent_state.items():
            if v == (row, col):
                return True
        return False

    def assign_goal(self, agent, goal):
        self.agent_goal[agent] = goal

    def step(self, agent, action):
        row, col = self.agent_state[agent]
        R, C = self.agent_map.shape
        if action == Action.RIGHT:
            s_prime = row, min(col + 1, C)
        elif action == Action.UP:
            s_prime = max(0, row - 1), col
        elif action == Action.LEFT:
            s_prime = row, max(0, col - 1)
        elif action == Action.DOWN:
            s_prime = min(row + 1, R), col
        elif action == Action.NOOP:
            s_prime = row, col
        else:
            raise ValueError("Invalid action.")
        if not self._occupied(*s_prime):
            self.agent_state[agent] = s_prime
        observation = self._observe(agent)
        reward = 0
        if self.agent_state[agent] == self.agent_goal[agent]:
            reward = 1
            self.agent_goal[agent] = None
        done = False
        return observation, reward, done, {}

    def reset(self):
        self.agent_state = {}
        self.agent_goal = {}
        for i, (row, col) in enumerate(zip(np.nonzero(self.agent_map))):
            self.agent_state[i] = (row, col)
            self.agent_goal[i] = None
        return self._observe()

    def render(self, mode="human"):
        pass

    def close(self):
        pass
