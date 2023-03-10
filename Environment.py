import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class State(Enum):
    START = 1
    BLANK = 2
    MINE = 3
    POWER = 4
    END = 5

class Action(Enum):
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4

class Policy(Enum):
    CLOSENESS = 1,
    MAXPOWER = 2

class Location:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Environment:
    def __init__(self, y_size = 5, x_size = 6, start_x = 0, start_y = 4, policy = Policy.CLOSENESS, alpha = 0.8, gamma = 0.95, random_states_distribution = False):
        states = [State.START, State.BLANK, State.MINE, State.POWER, State.END]
        actions = [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN]
        state_size = len(states)
        action_size = len(actions)
        self.q_table = np.zeros((state_size, action_size))
        self.alpha = alpha
        self.gamma = gamma
        self.policy = policy
        self.x_size = x_size
        self.y_size = y_size
        self.grid_size = y_size * x_size
        self.agent_location = Location(start_x, start_y)
        self.grid = np.zeros((self.y_size, self.x_size)).astype(int)
        
        if random_states_distribution:
            self.reset()
        else:
            self.hard_reset()

    def clear_grid(self):
        self.grid = np.zeros((self.y_size, self.x_size)).astype(int)
        return self.grid
            
    def reset(self):
        self.clear_grid()
        self.agent_location = Location(0, 0)
        # TODO: complete random reset

        

    def hard_reset(self):
        self.clear_grid()
        self.agent_location = Location(0, 0)
        self.grid[0, 0] = State.START.value
        self.grid[4, 4] = State.END.value
        self.grid[3, 0] = State.MINE.value
        self.grid[1, 1] = State.MINE.value
        self.grid[3, 3] = State.MINE.value
        self.grid[1, 4] = State.MINE.value
        self.grid[0, 2] = State.POWER.value
        self.grid[2, 2] = State.POWER.value
        self.grid[4, 1] = State.POWER.value
        self.grid[2, 5] = State.POWER.value


    def plot_grid(self):
        cmap = ListedColormap(['white', 'red', 'green'])
        plt.imshow(env.grid, cmap)
        plt.show()
        # TODO: solve color problem

    def step(state, action):
        pass

    def compute_reward():
        pass


env = Environment(random_states_distribution=False)
print(env.grid)
env.plot_grid()

    






