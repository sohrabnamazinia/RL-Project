import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import math

class State:
    def __init__(self, x, y, stateType):
        self.x = x
        self.y = y
        self.type = stateType

class StateType(Enum):
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

class Environment:
    def __init__(self, x_size = 5, y_size = 6, start_x = 0, start_y = 4, policy = Policy.CLOSENESS, alpha = 0.8, gamma = 0.95, mine_prob = 0.13, power_prob = 0.13, random_states_distribution = False):
        self.x_size = x_size
        self.y_size = y_size
        # grid contains state objects
        self.grid = np.zeros((self.x_size, self.y_size)).astype(State)
        self.actions = [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN]
        self.state_size = x_size * y_size
        self.action_size = len(self.actions)
        self.q_table = np.zeros((self.y_size, self.x_size, self.action_size)).astype(float)
        self.alpha = alpha
        self.gamma = gamma
        self.policy = policy
        self.mine_prob = mine_prob
        self.power_prob = power_prob
        # it will be initialized in the reset/hard_rest method
        self.end_state = None
        self.agent_state = None
        self.MAX_REWARD = 10000000
        self.MIN_REWARD = -10000000
        
        if random_states_distribution:
            self.reset()
        else:
            self.hard_reset()

    # sets all the grid as blank type
    def clear_grid(self):
        for i in range(self.x_size):
            for j in range(self.y_size):
                self.grid[i][j] = State(i, j, StateType.BLANK)
        return self.grid
            
    def reset(self):
        self.clear_grid()
        self.agent_state = State(random.randint(0, self.x_size), random.randint(0, self.y_size), StateType.BLANK)
        end_x = random.randint(0, self.x_size)
        end_y = random.randint(0, self.y_size)
        self.grid[end_x][end_y] = State(end_x, end_y, StateType.END)
        self.end_state = self.grid[end_x][end_y]
        for i in range(self.x_size):
            for j in range(self.y_size):
                if ((i, j) != (self.agent_state.x, self.agent_state.y) and (i, j) != (end_x, end_y)):
                    rand = random.random()
                    if (rand <= self.mine_prob):
                        self.grid[i][j] = State(i, j, StateType.MINE)
                    elif (rand <= self.mine_prob + self.power_prob):
                        self.grid[i][j] = State(i, j, StateType.POWER)
                    else:
                        self.grid[i][j] = State(i, j, StateType.BLANK)
        return self.grid            

    def hard_reset(self):
        self.clear_grid()
        self.agent_state = State(0, 0, StateType.BLANK)
        self.grid[0][2].type = StateType.POWER
        self.grid[2][2].type = StateType.POWER
        self.grid[2][5].type = StateType.POWER
        self.grid[4][1].type = StateType.POWER
        self.grid[1][1].type = StateType.MINE
        self.grid[1][4].type = StateType.MINE
        self.grid[3][0].type = StateType.MINE
        self.grid[3][3].type = StateType.MINE
        self.grid[4][4].type = StateType.END
        self.end_state = self.grid[4][4]
        return self.grid

    def convert_grid_to_digits(self):
        result = np.zeros((self.x_size, self.y_size))
        for i in range(self.x_size):
            for j in range(self.y_size):
                result[i][j] = self.grid[i][j].type.value
        return result

    def plot_grid(self):
        plt.imshow(env.convert_grid_to_digits(), cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.show()
        

    def print_grid(self):
        for i in range(self.x_size):
            for j in range(self.y_size):
                print(str(self.grid[i][j].type).split(".")[1], end=" ")
            print("\n")
        print("Starting Location of Agent: " + str(self.agent_state.x) + ", " + str(self.agent_state.y))

    def step(self, state, action):
        x, y = state.x, state.y
        if (action == Action.UP):
            x, y = max(x - 1, 0), y
        elif (action == Action.RIGHT):
            x, y = x, min(y + 1, self.y_size - 1)
        elif (action == Action.DOWN):
            x, y = min(x + 1, self.x_size - 1), y
        elif (action == Action.LEFT):
            x, y = x, max(y - 1, 0)
        else:
            print("Action not defined")
            return None

        next_state = self.grid[x][y] 
        reward = self.compute_reward(next_state)
        self.agent_state = next_state
        return (next_state, reward)
        

    def compute_reward(self, state_2):
        if state_2.type == StateType.END:
            return self.MAX_REWARD
        elif state_2.type == StateType.MINE:
            return self.MIN_REWARD
        elif (self.policy == Policy.CLOSENESS):
            return (1 / math.sqrt(((state_2.x - self.end_state.x)**2 + (state_2.y - self.end_state.y)**2)))
        elif (self.policy == Policy.MAXPOWER):
            if (state_2.type == StateType.POWER):
                return 1
            else:
                return 0


env = Environment(random_states_distribution=False)
env.print_grid()
#env.plot_grid()
state_1, reward_1 = env.step(env.agent_state, Action.DOWN)
print((env.agent_state.x, env.agent_state.y))
print("reward: " + str(reward_1))
state_2, reward_2 = env.step(env.agent_state, Action.DOWN)
print((env.agent_state.x, env.agent_state.y))
print("reward: " + str(reward_2))

