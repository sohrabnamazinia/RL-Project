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
    RIGHT = 0
    DOWN = 1


class Policy(Enum):
    CLOSENESS = 1,
    MAXPOWER = 2,
    COMBINATION = 3

class Environment:
    def __init__(self, x_size = 5, y_size = 6, x_start = 0, y_start = 0, policy = Policy.CLOSENESS, discount_factor = 0.95, mine_prob = 0.13, power_prob = 0.13, random_states_distribution = False, hard_reset_index = 1, has_mine = False):
        self.x_size = x_size
        self.y_size = y_size
        self.x_start = x_start
        self.y_start = y_start 
        # grid contains state objects
        self.grid = np.zeros((self.x_size, self.y_size)).astype(State)
        self.actions = [Action.RIGHT,Action.DOWN]
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
        elif hard_reset_index == 1:
            self.hard_reset_1(has_mine)
        elif hard_reset_index == 2:
            self.hard_reset_2()
        elif hard_reset_index == 3:
            self.hard_reset_3()
        else:
            print("Hard reset type not specified")

    # sets all the grid as blank type
    def clear_grid(self):
        for i in range(self.x_size):
            for j in range(self.y_size):
                self.grid[i][j] = State(i, j, StateType.BLANK)
        return self.grid
            
    def reset(self):
        self.clear_grid()
        self.agent_state = State(random.randint(0, self.x_size - 1), random.randint(0, self.y_size-1), StateType.BLANK)
        end_x = random.randint(0, self.x_size - 1)
        end_y = random.randint(0, self.y_size - 1)
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

    # original env (5*6)
    def hard_reset_1(self, has_mine):
        if has_mine:
            self.clear_grid()
            self.agent_state = self.grid[0][0]
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
        else:
            self.clear_grid()
            self.agent_state = self.grid[0][0]
            self.grid[0][2].type = StateType.POWER
            self.grid[2][2].type = StateType.POWER
            self.grid[2][5].type = StateType.POWER
            self.grid[4][1].type = StateType.POWER
            self.grid[4][4].type = StateType.END
            self.end_state = self.grid[4][4]
            return self.grid

    # 10*10 env
    def hard_reset_2(self):
        self.clear_grid()
        self.agent_state = self.grid[0][0]
        self.grid[0][2].type = StateType.POWER
        self.grid[0][4].type = StateType.POWER
        self.grid[0][6].type = StateType.POWER
        self.grid[0][8].type = StateType.POWER
        self.grid[1][4].type = StateType.POWER
        self.grid[2][5].type = StateType.POWER
        self.grid[4][2].type = StateType.POWER
        self.grid[6][4].type = StateType.POWER
        self.grid[7][7].type = StateType.POWER
        self.grid[9][9].type = StateType.END
        self.end_state = self.grid[9][9]
        return self.grid
    
    # 100*100 env
    def hard_reset_3(self):
        self.clear_grid()
        self.agent_state = self.grid[0][0]
        for i in range(self.x_size):
           self.grid[i][i].type = StateType.POWER 
        self.grid[0][0].type = StateType.BLANK
        self.grid[99][99].type = StateType.END 
        self.end_state = self.grid[99][99]
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
        if (action == Action.RIGHT):
            x, y = x, min(y + 1, self.y_size - 1)
        elif (action == Action.DOWN):
            x, y = min(x + 1, self.x_size - 1), y
        else:
            print("Action not defined")
            x, y = float("inf"), float("inf")

        if x != float("inf") and y != float("inf"):
            next_state = self.grid[x][y]
        else:
            next_state = None
        reward = self.compute_reward(state, next_state, self.policy, visited_power_states)
        self.agent_state = next_state
        return (next_state, reward)
        
    def manhattan_dist(self, state1, state2):
        #term1 = math.pow(state1.x - state2.x, 2)
        #term2 = math.pow(state1.y - state2.y, 2)
        
        term1 = abs(state1.x - state2.x)
        term2 = abs(state1.y - state2.y)
        return (term1 + term2)
        #return math.sqrt(term1 + term2)

    def compute_reward(self, state_1, state_2, policy, visited_power_states):

        if state_2 == None:
            return 1000 * self.MIN_REWARD
        if policy == Policy.CLOSENESS:
            return self.compute_reward_closeness(state_1, state_2)
        
        elif policy == Policy.MAXPOWER:
            return self.compute_reward_maxpower(state_2, visited_power_states)

        elif policy == Policy.COMBINATION:
            return self.compute_reward_combination(state_1, state_2, visited_power_states)
        else:
            print("Policy not defined")
            return None
    
    def compute_reward_closeness(self, state_1, state_2):
        if state_2.type == StateType.END:
            return self.MAX_REWARD
        elif state_2.type == StateType.MINE:
            return self.MIN_REWARD
        else:
            return self.manhattan_dist(state_1, self.end_state) - self.manhattan_dist(state_2, self.end_state)

    def compute_reward_maxpower(self, state_2, visited_power_states):
        if state_2.type == StateType.END:
            return self.MAX_REWARD
        elif state_2.type == StateType.MINE:
            return self.MIN_REWARD
        else:
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
    
    def compute_reward_combination(self, state_1, state_2, visited_power_states):
        r1 = self.compute_reward_closeness(state_1, state_2)
        r2 = self.compute_reward_maxpower(state_2, visited_power_states)
        return r1 + r2


    # return True if the current state is the goal state
    def reached_goal(self):
        if self.agent_state.type == StateType.END:
            return True
        return False

    def compute_reward_ground_truth_path(self, path):
        reward = 0
        # init current state with the start state
        current_state = self.grid[self.x_start][self.y_start]
        visited_power_states = []
        if (self.grid[current_state.x][current_state.y].type == StateType.POWER):
                visited_power_states.append(current_state)
        for action in path:
            next_state, r = self.step(current_state, action, [])
            reward += r
            current_state = next_state
            if (self.grid[current_state.x][current_state.y].type == StateType.POWER):
                visited_power_states.append(current_state)
        return reward

    @staticmethod
    def getActionFromStateTuples(state1, state2):

        x1, y1 = state1[0], state1[1]
        x2, y2 = state2[0], state2[1]
        action = None

        if x1 == x2 and y1 + 1 == y2:
            action = Action.RIGHT
        elif x1 + 1 == x2 and y1 == y2:
            action = Action.DOWN

        return action

    def compute_reward_dag_paths(self, paths, ground_truth_reward, print_rewards=False):
        rewards = []
        counter = 0
        for path in paths:
            reward = 0
            visited_power_states = []
            for i in range(len(path) - 1):
                state_tuple_0, state_tuple_1 = path[i], path[i + 1]
                action = Environment.getActionFromStateTuples(state_tuple_0, state_tuple_1)
                state_0 = self.grid[state_tuple_0[0]][state_tuple_0[1]]
                state_1, r = self.step(state_0, action, [])
                reward += r
                if (state_0.type == StateType.POWER):
                    visited_power_states.append(state_0)
            # this is for debugging        
            if (reward > ground_truth_reward):
                actions = []
                for i in range(len(path) - 1):
                    actions.append(str(self.getActionFromStateTuples(path[i], path[i + 1])).split(".")[1])
                print("Path index: " + str(counter) + ", with cumulative reward: " + str(reward) + "\n" + str(actions))
            counter += 1
            rewards.append(reward)
        return rewards
    
    def get_max_reward_brute_force_path(self, paths):
        max_reward = -math.inf
        max_reward_path = None
        for path in paths:
            reward = 0
            visited_power_states = []
            for i in range(len(path) - 1):
                state_tuple_0, state_tuple_1 = path[i], path[i + 1]
                action = Environment.getActionFromStateTuples(state_tuple_0, state_tuple_1)
                state_0 = self.grid[state_tuple_0[0]][state_tuple_0[1]]
                state_1, r = self.step(state_0, action, [])
                reward += r
                if (state_0.type == StateType.POWER):
                    visited_power_states.append(state_0)
            if reward > max_reward:
                max_reward = reward
                max_reward_path = path
        return max_reward_path
            
