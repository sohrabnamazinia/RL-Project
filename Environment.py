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
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class Policy(Enum):
    CLOSENESS = 1,
    MAXPOWER = 2

class Environment:
    def __init__(self, x_size = 5, y_size = 6, x_start = 0, y_start = 0, policy = Policy.CLOSENESS, discount_factor = 0.95, mine_prob = 0.13, power_prob = 0.13, random_states_distribution = False):
        self.x_size = x_size
        self.y_size = y_size
        self.x_start = x_start
        self.y_start = y_start 
        # grid contains state objects
        self.grid = np.zeros((self.x_size, self.y_size)).astype(State)
        self.actions = [Action.UP, Action.RIGHT,Action.DOWN, Action.LEFT]
        self.state_size = x_size * y_size
        self.action_size = len(self.actions)
        self.q_table = np.zeros((self.x_size, self.y_size, self.action_size)).astype(float)
        self.discount_factor = discount_factor
        self.policy = policy
        self.mine_prob = mine_prob
        self.power_prob = power_prob
        # it will be initialized in the reset/hard_rest method
        self.end_state = None
        self.agent_state = None
        self.MAX_REWARD = 1000
        self.MIN_REWARD = -1000000000000
        #self.POWER_REWARD = 90
        
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
        self.agent_state = State(random.randint(0, self.x_size - 1), random.randint(0, self.y_size-1), StateType.BLANK)
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
        plt.imshow(self.convert_grid_to_digits(), cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.show()
        

    def print_grid(self):
        for i in range(self.x_size):
            for j in range(self.y_size):
                print(str(self.grid[i][j].type).split(".")[1], end=" ")
            print("\n")
        print("Current Location of Agent: " + str(self.agent_state.x) + ", " + str(self.agent_state.y) + "\n")

    def step(self, state, action, visited_power_states):
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
        reward = self.compute_reward(state, next_state, self.policy, visited_power_states)
        self.agent_state = next_state
        return (next_state, reward)
        
    def euclidean_dist(self, state1, state2):
        term1 = math.pow(state1.x - state2.x, 2)
        term2 = math.pow(state1.y - state2.y, 2)
        return math.sqrt(term1 + term2)

    def compute_reward(self, state_1, state_2, policy, visited_power_states):
        if state_2.type == StateType.END:
                return self.MAX_REWARD
        elif state_2.type == StateType.MINE:
                return self.MIN_REWARD
        elif policy == Policy.CLOSENESS:
                return self.euclidean_dist(state_1, self.end_state) - self.euclidean_dist(state_2, self.end_state); 
        elif policy == Policy.MAXPOWER:
            # if self.grid[state_2.x][state_2.y].type == StateType.POWER:
            #     return self.POWER_REWARD
            # else:
            #     return 0
            reward = 0
            region = []
            x = state_2.x
            y = state_2.y
            for i in range(-2, 3):
                for j in range(-2, 3):
                    candidate = self.grid[min(max(x + i, 0), self.x_size - 1)][min(max(y + j, 0), self.y_size - 1)]
                    if (candidate not in region):
                        region.append(candidate)
            for candidate in region:
                if (candidate.type == StateType.POWER) and (candidate not in visited_power_states):
                    reward += 1
            return reward

        else:
            print("Policy not defined")
            return None
             
    
    # return True if the current state is the goal state
    def reached_goal(self):
        if self.agent_state.type == StateType.END:
            return True
        return False

