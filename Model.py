import numpy as np
import random
from Environment import Environment 
from Environment import Action 
from operator import itemgetter

class Model:
    def __init__(self, env, exploration_prob = 1, exploration_decreasing_rate = 0.001, min_exploration_prob = 0.1, max_iter_per_episode = 100, episode_count = 1000):
        self.environment = env
        self.exploration_prob = exploration_prob
        self.exploration_decreasing_rate = exploration_decreasing_rate
        self.min_exploration_prob = min_exploration_prob
        self.max_iter_per_episode = max_iter_per_episode
        self.episode_count = episode_count
    
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
        rewards_per_episode = []
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
            rewards_per_episode.append(total_episode_reward)
        return rewards_per_episode

    
    def test(self):
        pass

