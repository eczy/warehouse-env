import gym
from gym import spaces
import numpy as np
from enum import Enum
from PIL import Image
from matplotlib import cm
from od_mstar3 import cpp_mstar
import networkx as nx

class Action(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3
    NOOP = 4


class WarehouseEnv(gym.Env):
    def __init__(self, obstacle_map, agent_map, 
                 max_timestep=None, 
                 render_as_observation=False, 
                 coordinated_planner=False,
                 local_obseration_size=(11,11)):
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
        self.num_agents = np.count_nonzero(self.agent_map)
        
        self.local_observation_width = int(np.floor(local_obseration_size[0]/2.0))
        self.local_observation_length = int(np.floor(local_obseration_size[1]/2.0))
        self.local_obs_shape = local_obseration_size
        
        self.render_as_observation = render_as_observation
        self.zoom_observation_size = 4
        
        self.env_graph = self.warehouse_to_graph()
        
        self.toll_map = {} #np.ones_like(agent_map) * 10.0
        for edge in self.env_graph.edges:
            e1, e2 = edge[0], edge[1]
            self.toll_map[edge] = 10.0
            
        self.update_network_edges()
        
        self.R = 1e-5 # np.ones(self.obs_shape) * 1e-4
        self.beta = 4.0 * self.num_agents # np.ones(self.obs_shape) * 4
        
        self.coordinated_planner = coordinated_planner
        
        if self.render_as_observation:
            self.obs_shape = np.array(self.render(zoom_size=self.zoom_observation_size, local=True))[:,:,:-1].shape
            self.observation_space = spaces.Box(low=0, high=255, shape=self.obs_shape, dtype=np.uint8)
        else:
#             local_agent_state, local_agent_goal, local_obstacles = \
#                             self.local_observation(agent, self.local_obs_shape)
            self.obs_shape = [local_obseration_size, local_obseration_size, 5]
            self.observation_space = spaces.Box(low=0, high=255, shape=self.obs_shape, dtype=np.uint8)
            

    def warehouse_to_graph(self):
        env = self
        w, h = env.agent_map.shape[0], env.agent_map.shape[1]
        G = nx.grid_2d_graph(w, h)
        H = G.to_directed()

        obstacles = np.argwhere(env.obstacle_map > 0)

        _ = [H.remove_node((o[0], o[1])) for o in obstacles]
        return H
    
    def update_network_edges(self):
        for edge in self.env_graph.edges:
            e1, e2 = edge[0], edge[1]
            self.env_graph[e1][e2]['weight'] = self.toll_map[edge]
    
    def get_local_goal_for_agent(self, agent_id, x_range, y_range, coordinated_planner=False):
        env = self
        env_graph = self.env_graph
        
        min_x, max_x = x_range
        min_y, max_y = y_range
        if coordinated_planner:
            states = [(v[0], v[1]) for k, v in env.agent_state.items()]
            goals = [(v[0], v[1]) for k, v in env.agent_goal.items()]
            path = None
            start_x, start_y = None, None
            next_x, next_y = None, None

            try:
                path = cpp_mstar.find_path(env.obstacle_map, states, goals, 10, 5 * 60.0)

                start_x, start_y = env.agent_state[agent_id]
                next_x, next_y = path[1][agent_id]
                
                inside_range = True
                counter = 1
                while (inside_range):
                    current_x, current_y = path[counter][agent_id]
                    if (current_x < min_x) or (current_x > max_x) or \
                        (current_y < min_y) or (current_y > max_y):
                        inside_range = False
                        return path[counter-1][agent_id]
                        break
                    counter += 1
            except:
                coordinated_planner = False

        if not coordinated_planner:
            location = env.agent_state[agent_id]
            goal = env.agent_goal[agent_id]

            path = nx.astar_path(env_graph, (location[0], location[1]), (goal[0], goal[1]), weight='weight')

            start_x, start_y = path[0]
            next_x, next_y = path[1]
            
            inside_range = True
            counter = 1
            while (inside_range):
                current_x, current_y = path[counter]
                if (current_x < min_x) or (current_x >= max_x) or \
                    (current_y < min_y) or (current_y >= max_y):
                    inside_range = False
                    return path[counter-1]
                    break
                counter += 1
    
    def local_observation(self, agent_id, local_obs_shape):
        x, y = self.agent_state[agent_id]
        h, w = local_obs_shape
        R, C = self.agent_map.shape
        R = R - 1
        C = C - 1
        x_start, x_end, y_start, y_end = 0, 0, 0, 0
        
        if x + np.floor(w / 2.0) > R: 
            x_start = int(np.floor(R + 1 - w)) 
            x_end = R + 1
        elif x - np.floor(w / 2.0) < 0: 
            x_start = 0
            x_end = int(np.floor(w))
        else:
            x_start = int((x - np.floor(w / 2.0)))
            x_end = int((x + np.floor(w / 2.0)) + 1)
        
        if (y + np.floor(h / 2.0)) > C: 
            y_start = int(np.floor(C + 1 - h))
            y_end = C + 1
        elif (y - np.floor(h / 2.0)) < 0: 
            y_start = 0
            y_end = int(np.floor(h))
        else:
            y_start = int((y - np.floor(h / 2.0)))
            y_end = int((y + np.floor(h / 2.0)) + 1) 
        
        x_start = int(np.maximum(0, x_start))
        y_start = int(np.maximum(0, y_start))
        
        local_agent_state = {}
        local_agent_goal = {}
        local_obstacles = self.obstacle_map[x_start:x_end, y_start:y_end]
        for agent_id, agent_loc in self.agent_state.items():
            agent_x, agent_y = agent_loc
            if (agent_x >= x_start) and (agent_x < x_end) and \
                    (agent_y >= y_start) and (agent_y < y_end): 
                #get local goal if its out of view
                goal_x, goal_y = self.agent_goal[agent_id]
                
                if (goal_x < x_start) or (goal_x >= x_end) or \
                    (goal_y < y_start) or (goal_y >= y_end):
                    goal_x, goal_y = self.get_local_goal_for_agent(agent_id, 
                                                  (x_start, x_end), (y_start, y_end),
                                                  coordinated_planner=self.coordinated_planner)
                
                local_agent_state[agent_id] = (agent_x - x_start, agent_y - y_start)
                local_agent_goal[agent_id] = (goal_x - x_start, goal_y - y_start)
                
        return local_agent_state, local_agent_goal, local_obstacles
        
      
    def _observe(self, agent_id=None):
        
        if agent_id is None:
            agent = self.current_agent_id
        else:
            agent = agent_id
        
        if self.render_as_observation:
            im_output = self.render(zoom_size=4, agent_id=agent, local=True)
            return np.array(self.render(zoom_size=4, agent_id=agent))[:,:,:-1]
        
        local_agent_state, local_agent_goal, local_obstacles = self.local_observation(agent, self.local_obs_shape)
        
        goal = local_agent_goal[agent]
        state = local_agent_state[agent]
        
        agent_channel = np.zeros_like(local_agent_map)
        agent_channel[state] = 1
        other_agent_channel = np.zeros_like(local_agent_map)
        current_key = 1
        for k, v in local_agent_state.items():
            if k != agent:
                current_key += 1
                other_agent_channel[v] = current_key
        
        goal_channel = np.zeros_like(local_goal_map)
        goal_channel[goal] = 1
        other_goal_channel = np.zeros_like(local_goal_map)
        current_key = 1
        for k, v in local_agent_goal.items():
            if k != agent:
                current_key += 1
                other_goal_channel[v] += current_key
        
        return np.stack(
            [agent_channel, other_agent_channel, goal_channel, other_goal_channel, local_obstacles],
            axis=0,
        ).transpose((1, 2, 0))
    

    def _occupied(self, row, col):
        if self.obstacle_map[row, col] != 0:
            return True
        for _, v in self.agent_state.items():
            if v == (row, col):
                return True
        return False

    def assign_goal(self, agent, goal):
        if self._occupied(goal[0], goal[1]):
            raise ValueError("Attempting to assign goal to occupied location: {}, {}.".format(agent, goal))
        self.agent_goal[agent] = goal

    def step(self, action, agent_id=None):
        if agent_id is None:
            agent = self.current_agent_id
        else:
            agent = agent_id
            
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
        
        old_loc = self.agent_state[agent]
        new_loc = s_prime
        if not self._occupied(*s_prime):
            self.agent_state[agent] = s_prime
            delta = 0.0
        else:
            delta = 1.0
            reward = -0.2
        
        edge = ((new_loc[0], new_loc[1]), (old_loc[0], old_loc[1]))
        self.toll_map[edge] =  self.R * self.beta * delta + (1 - self.R) * self.toll_map[edge]
        
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
        
        observation = self._observe(agent_id=agent)
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

    def render(self, mode="human", zoom_size=4, agent_id=None, local=True):
        if local:
            if agent_id is None:
                agent_id = self.current_agent_id
            local_agent_state, local_agent_goal, local_obstacles = \
                            self.local_observation(agent_id, self.local_obs_shape)
            agent_state_map = local_agent_state
            agent_goal_map = local_agent_goal
            obstacles = local_obstacles
        else:
            agent_state_map = self.agent_state
            agent_goal_map = self.agent_goal
            obstacles = self.obstacle_map
                
        image_array = np.zeros_like(obstacles)
        
        agent_color_map = {}
        if agent_id is not None:
            agent_color_map[agent_goal_map[agent_id]] = 1

        current_color_map_counter = 2
        for agent, agent_loc in agent_state_map.items():
            assigned_color = None
            agent_goal_loc = agent_goal_map[agent]
            if agent_goal_loc in agent_color_map:
                assigned_color = agent_color_map[agent_goal_loc]
                image_array[agent_loc] = assigned_color
            else:
                agent_color_map[agent_goal_loc] = current_color_map_counter
                assigned_color = current_color_map_counter
                image_array[agent_loc] = assigned_color
                current_color_map_counter += 1
        
        image_array = np.where(obstacles != 0, current_color_map_counter + 3, image_array)

        # Color obstacles and "zoom in" by repeating appropiately
        max_agent_id = current_color_map_counter
        image_array_copy = np.repeat(image_array, zoom_size, axis=0)
        image_array_copy2 = np.repeat(image_array_copy, zoom_size, axis=1)

        # Set inner goal boxes for each agent
        inner_box_size = int(zoom_size/4)
        outer_box_size = int(zoom_size/3)
        smaller_box_size = int(zoom_size/5)
        for k, v in agent_goal_map.items():
            x = int((v[0] * zoom_size) + zoom_size/2) 
            y = int((v[1] * zoom_size) + zoom_size/2) 
            
            #Set goal outer color
            image_array_copy2[(x-outer_box_size):(x+outer_box_size), 
                              (y-outer_box_size):(y+outer_box_size)] = max_agent_id + 4
            
            #Set goal inner color
            image_array_copy2[(x-inner_box_size):(x+inner_box_size), 
                              (y-inner_box_size):(y+inner_box_size)] = agent_color_map[v]
            
            if (agent_id is not None) and (k == agent_id):
                agent_x, agent_y = agent_state_map[agent_id]
                agent_x = int((agent_x * zoom_size) + zoom_size/2) 
                agent_y = int((agent_y * zoom_size) + zoom_size/2) 
                image_array_copy2[(agent_x-smaller_box_size):(agent_x+smaller_box_size), 
                                  (agent_y-smaller_box_size):(agent_y+smaller_box_size)] = max_agent_id + 5

        scalar_cm = cm.ScalarMappable(cmap="jet_r")
        color_array = np.uint8(scalar_cm.to_rgba(image_array_copy2)*255)

        #color background gray and obstacles black
        color_array[(image_array_copy2 == 0)] = [190,190,190,255]
        color_array[(image_array_copy2 == (max_agent_id + 3))] = [0,0,0,255]
        color_array[(image_array_copy2 == (max_agent_id + 4))] = [255,255,255,255]
        color_array[(image_array_copy2 == (max_agent_id + 5))] = [255,255,255,255]

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