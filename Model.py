import numpy as np
import random
from Environment import Environment 
from Environment import Action 
from operator import itemgetter

class Model:
    def __init__(self, env, exploration_prob = 1, exploration_decreasing_rate = 0.0001, min_exploration_prob = 0.1, max_iter_per_episode = 100, episode_count = 10000):
        self.environment = env
        self.exploration_prob = exploration_prob
        self.exploration_decreasing_rate = exploration_decreasing_rate
        self.min_exploration_prob = min_exploration_prob
        self.max_iter_per_episode = max_iter_per_episode
        self.episode_count = episode_count
        self.current_optimal_policy = np.zeros((self.environment.x_size, self.environment.y_size)).astype(Action)
        self.rewards_per_episode = []
    
    def select_action(self):
        current_x = self.environment.agent_state.x
        current_y = self.environment.agent_state.y
        Qs = self.environment.q_table[current_x][current_y]
        action = self.get_best_action(Qs)
        return action

    def get_best_action(self, Qs):
        index, element = max(enumerate(Qs), key=itemgetter(1))
        if index == 0:
            return Action.UP
        elif index == 1:
            return Action.RIGHT
        elif index == 2:
            return Action.DOWN
        elif index == 3:
            return Action.LEFT
        else:
            print("Action not defined")
            return None  
    
    def select_random_action(self):
        action_index = random.randint(0, 3)
        if (action_index == 0):
            return Action.UP
        elif (action_index == 1):
            return Action.RIGHT
        elif (action_index == 2):
            return Action.DOWN
        elif (action_index == 3):
            return Action.LEFT
        else:
            print("Action not defined")
            return None
        
    
    def train(self):
        for i in range(self.episode_count):
            total_episode_reward = 0
            for j in range(self.max_iter_per_episode):
                action = None
                if (random.random() <= self.exploration_prob):
                    action = self.select_random_action()
                else:
                    action = self.select_action()
                old_state = self.environment.agent_state
                next_state, reward = self.environment.step(self.environment.agent_state, action)
                self.environment.q_table[old_state.x][old_state.y][action.value] = (1 - self.environment.lr) * self.environment.q_table[old_state.x][old_state.y][action.value] + self.environment.lr * (reward + self.environment.discount_factor * (max(self.environment.q_table[next_state.x][next_state.y])))
                total_episode_reward += reward
                if (self.environment.reached_goal()):
                    break
                # exploration-exploitation tradeoff: linear decay implementaiton
                self.exploration_prob = max(self.exploration_prob - self.exploration_decreasing_rate, self.min_exploration_prob)
            self.rewards_per_episode.append(total_episode_reward)
        return self.get_current_optimal_policy()

    def get_current_optimal_policy(self):
        for i in range(self.environment.x_size):
            for j in range(self.environment.y_size):
                q_values = self.environment.q_table[i][j]
                index, element = max(enumerate(q_values), key=itemgetter(1))
                self.current_optimal_policy[i][j] = self.environment.actions[index]
        return self.current_optimal_policy
    
    def print_current_optimal_policy(self):
        for i in range(self.environment.x_size):
            for j in range(self.environment.y_size):
                print(str(self.current_optimal_policy[i][j]).split(".")[1], end=" ")
            print("\n")
    
    def test(self, print_result=True):
        if (print_result):
            self.print_current_optimal_policy()
        reachedGoal = False
        x = self.environment.x_start
        y = self.environment.y_start
        path = []
        while (not reachedGoal):
            action = self.current_optimal_policy[x][y]
            path.append(action)
            if (action == Action.UP):
                x, y = max(x - 1, 0), y
            elif (action == Action.RIGHT):
                x, y = x, min(y + 1, self.environment.y_size - 1)
            elif (action == Action.DOWN):
                x, y = min(x + 1, self.environment.x_size - 1), y
            elif (action == Action.LEFT):
                x, y = x, max(y - 1, 0)
            else:
                print("Action not defined")
            
            if ((x, y) == (self.environment.end_state.x, self.environment.end_state.y)):
                reachedGoal = True

        print("OPTIMAL PATH:")    
        for action in path:
            print(str(action).split(".")[1], end=" ")

        return self.current_optimal_policy

