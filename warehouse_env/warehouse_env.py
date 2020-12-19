import gym
from gym import spaces
import numpy as np
from enum import Enum
from PIL import Image
from matplotlib import cm

class Action(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3
    NOOP = 4


class WarehouseEnv(gym.Env):
    def __init__(self, obstacle_map, agent_map, max_timestep=None):
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
        
        action_space_map = {}
        action_space_map[0] = Action.RIGHT
        action_space_map[1] = Action.UP
        action_space_map[2] = Action.LEFT
        action_space_map[3] = Action.DOWN
        action_space_map[4] = Action.NOOP
        self.action_space_map = action_space_map
        
        self.max_timestep = max_timestep
        self.timestep = 0
        
        for agent_id in self.agent_goal.keys():
            self.assign_goal(agent_id, self.get_new_goal_location())
            
        self.current_agent_id = 0
        self.num_agents = np.unique(self.agent_map).shape[0] - 1
        
        self.obs_shape = [self.agent_map.shape[0], self.agent_map.shape[1], 5]
        self.observation_space = spaces.Box(low=0, high=255, shape=self.obs_shape, dtype=np.uint8)

    def _observe(self, agent_id=None):
        if agent_id is None:
            agent = self.current_agent_id
        else:
            agent = agent_id
        
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
            [agent_channel, other_agent_channel, goal_channel, other_goal_channel, self.obstacle_map],
            axis=0,
        ).transpose((1, 2, 0))
    
#         agent_channel = np.zeros(self.obs_shape)
#         agent_channel[state] = 1
#         other_agent_channel = np.zeros(self.obs_shape)
#         for k, v in self.agent_state.items():
#             if k != agent:
#                 other_agent_channel[v] = k+2
        
#         agent_channel[goal] = 1
#         for k, v in self.agent_goal.items():
#             if k != agent:
#                 other_agent_channel[v[0], v[1]+1] = k+2
        
#         return np.stack(
#             [agent_channel, other_agent_channel, self.obstacle_map],
#             axis=0,
#         ).transpose((1, 2, 0))

    def _occupied(self, row, col):
        if self.obstacle_map[row, col] != 0:
            return True
        for _, v in self.agent_state.items():
            if v == (row, col):
                return True
        return False

    def assign_goal(self, agent, goal):
        if self._occupied(goal[0], goal[1]):
            raise ValueError("Attempting to assgin goal to occupied location: {}, {}.".format(agent, goal))
        self.agent_goal[agent] = goal

    def step(self, action, agent_id=None):
        if agent_id is None:
            agent = self.current_agent_id
            
#         action = self.action_space_map[action] if isinstance(action, int) else action
#         print(action, action == Action.NOOP, action == Action.RIGHT, 
#               action == Action.LEFT, action == Action.UP, action == Action.DOWN)
        row, col = self.agent_state[agent]
        R, C = self.agent_map.shape
        R = R-1
        C = C-1
        if action == 0:
            s_prime = row, min(col + 1, C)
        elif action == 1:
            s_prime = max(0, row - 1), col
        elif action == 2:
            s_prime = row, max(0, col - 1)
        elif action == 3:
            s_prime = min(row + 1, R), col
        elif action == 4:
            s_prime = row, col
        else:
            raise ValueError("Invalid action.")
        if not self._occupied(*s_prime):
            self.agent_state[agent] = s_prime
        else:
            reward = -0.2
        observation = self._observe(agent_id=agent)
        reward = 0
        if self.agent_state[agent] == self.agent_goal[agent]:
            reward = 5
            # Lazy retry, fix me
            new_goal_location_occupied = True
            while_count = 0
            while new_goal_location_occupied:
                while_count += 1
                new_goal_location = self.get_new_goal_location(excluding_location=self.agent_goal[agent])
                if not self._occupied(new_goal_location[0], new_goal_location[1]):
                    new_goal_location_occupied = False
                if while_count > 800:
                    raise ValueError("Probably in an infinite while loop.")
                
            self.assign_goal(agent, new_goal_location)
        
        self.timestep += 1
        self.current_agent_id = (self.current_agent_id + 1) % self.num_agents
        
        if self.max_timestep is None:
            done = False
        else:
            done = True if self.timestep <= self.max_timestep else False
        return observation, reward, done, {}
    
    def get_new_goal_location(self, excluding_location=None):
        obstacle_map_copy = self.obstacle_map.copy()
        if excluding_location is not None:
            obstacle_map_copy[excluding_location] = 1
        empty_locations = np.argwhere((self.agent_map == 0) & (obstacle_map_copy == 0))
        choice_location = np.random.choice(empty_locations.shape[0])
        x, y = empty_locations[choice_location]
        return (x, y)
        
    def reset(self):
        self.agent_state = {}
        self.agent_goal = {}
        rows, cols = np.nonzero(self.agent_map)
        for i, (row, col) in enumerate(zip(rows, cols)):
            self.agent_state[i] = (row, col)
            self.agent_goal[i] = None
            
        for agent_id in self.agent_goal.keys():
            self.assign_goal(agent_id, self.get_new_goal_location())
            
        self.current_agent_id = 0
        return self._observe()

    def render(self, mode="human", zoom_size=8, agent_id=None, other_agents_same=False):
        image_array = np.zeros_like(self.agent_map)
        # Map agents, agent_id will mapped to 1
        for k, v in self.agent_state.items():
            if agent_id is not None:
                if agent_id == k:
                    image_array[v] = 1
                else:
                    if not other_agents_same:
                        image_array[v] = k+2
                    else:
                        image_array[v] = 2
            else:
                image_array[v] = k+1
        
        # Color obstacles and "zoom in" by repeating appropiately
        max_agent_id = max(self.agent_goal, key=int)
        image_array = np.where(self.obstacle_map != 0, max_agent_id + 3, image_array)
        image_array_copy = np.repeat(image_array, zoom_size, axis=0)
        image_array_copy2 = np.repeat(image_array_copy, zoom_size, axis=1)
        
        # Set inner goal boxes for each agent
        inner_box_size = int(zoom_size/4)
        outer_box_size = int(zoom_size/3)
        for k, v in self.agent_goal.items():
            x = int((v[0] * zoom_size) + zoom_size/2) 
            y = int((v[1] * zoom_size) + zoom_size/2) 

            image_array_copy2[(x-outer_box_size):(x+outer_box_size), 
                              (y-outer_box_size):(y+outer_box_size)] = max_agent_id + 4
            if agent_id is not None:
                if agent_id == k:
                    image_array_copy2[(x-inner_box_size):(x+inner_box_size), 
                                      (y-inner_box_size):(y+inner_box_size)] = 1
                else:
                    if not other_agents_same:
                        image_array_copy2[(x-inner_box_size):(x+inner_box_size), 
                                          (y-inner_box_size):(y+inner_box_size)] = k+2
                    else:
                        image_array_copy2[(x-inner_box_size):(x+inner_box_size), 
                                          (y-inner_box_size):(y+inner_box_size)] = 2
            else:
                image_array_copy2[(x-inner_box_size):(x+inner_box_size), 
                                  (y-inner_box_size):(y+inner_box_size)] = k+1
        
        scalar_cm = cm.ScalarMappable(cmap="jet_r")
        color_array = np.uint8(scalar_cm.to_rgba(image_array_copy2)*255)

        #color background gray and obstacles black
        color_array[(image_array_copy2 == 0)] = [190,190,190,255]
        color_array[(image_array_copy2 == (max_agent_id + 3))] = [0,0,0,255]
        color_array[(image_array_copy2 == (max_agent_id + 4))] = [255,255,255,255]
        if agent_id is not None:
            color_array[(image_array_copy2 == 1)] = [255,0,0,255]

        im = Image.fromarray(color_array)
        return im
    
    def close(self):
        pass

class WarehouseEnvRuntimeError(Exception):
    def __init__(self, message, errors):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self.errors = errors
        
        raise errors