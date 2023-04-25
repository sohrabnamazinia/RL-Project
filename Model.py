import numpy as np
import math
import random
from Environment import Environment, StateType 
from Environment import Action 
from operator import itemgetter

class Model:
    def __init__(self, env, exploration_prob = 1, learning_rate = 0.8, exploration_decreasing_rate = 0.0001, learning_rate_decreasing_rate = 0.0001, min_exploration_prob = 0.1, min_learning_rate = 0.1, max_iter_per_episode = 1000, episode_count = 10000):
        self.environment = env
        self.exploration_prob = exploration_prob
        self.exploration_decreasing_rate = exploration_decreasing_rate
        self.learning_rate_decreasing_rate = learning_rate_decreasing_rate
        self.min_exploration_prob = min_exploration_prob
        self.max_iter_per_episode = max_iter_per_episode
        self.episode_count = episode_count
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.current_optimal_policy = np.zeros((self.environment.x_size, self.environment.y_size)).astype(Action)
        self.rewards_per_episode = []
    
    def apply_boundry_move_constraint(self, Qs, agent_state):
        if (agent_state.x == 0):
            Qs[0] = -math.inf
        if (agent_state.y == 0):
            Qs[3] = -math.inf
        if (agent_state.x == self.environment.x_size - 1):
            Qs[1] = -math.inf
        if (agent_state.y == self.environment.y_size - 1):
            Qs[2] = -math.inf
        return Qs

    def select_action(self, agent_state):
        current_x = self.environment.agent_state.x
        current_y = self.environment.agent_state.y
        Qs = self.environment.q_table[current_x][current_y]
        action = self.get_best_action(Qs, agent_state)
        return action

    def get_best_action(self, Qs, agent_state):
        Qs = self.apply_boundry_move_constraint(Qs, agent_state)
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
    
    def select_random_action(self, agent_state):
        possible_actions = self.environment.actions.copy()
        if (agent_state.x == 0):
            possible_actions.remove(self.environment.actions[0])
        if (agent_state.x == self.environment.x_size - 1):
            possible_actions.remove(self.environment.actions[2])
        if (agent_state.y == 0):
            possible_actions.remove(self.environment.actions[3])
        if (agent_state.y == self.environment.y_size - 1):
            possible_actions.remove(self.environment.actions[1])
        if len(possible_actions) == 0:
            print("No Possible Action!")
            return None
        else:
            action = random.choice(possible_actions)
            return action
        
    
    def train(self):
        for i in range(self.episode_count):
            total_episode_reward = 0
            # keep track of visited states
            visited_power_states = []
            # reset agent to some random location
            self.environment.agent_state = self.environment.grid[random.randint(0, self.environment.x_size - 1)][random.randint(0, self.environment.y_size - 1)]
            for j in range(self.max_iter_per_episode):
                action = None
                if (random.random() <= self.exploration_prob):
                    action = self.select_random_action(self.environment.agent_state)
                else:
                    action = self.select_action(self.environment.agent_state)

                old_state = self.environment.agent_state
                # update visited_power_states
                if (old_state.type == StateType.POWER):
                    visited_power_states.append(old_state)

                next_state, reward = self.environment.step(self.environment.agent_state, action, visited_power_states)
                self.environment.q_table[old_state.x][old_state.y][action.value] = (1 - self.learning_rate) * self.environment.q_table[old_state.x][old_state.y][action.value] + self.learning_rate * (reward + self.environment.discount_factor * (max(self.environment.q_table[next_state.x][next_state.y])))
                total_episode_reward += reward
                if (self.environment.reached_goal()):
                    break
                # exploration-exploitation tradeoff: linear decay implementaiton
                self.learning_rate = max(self.learning_rate - self.learning_rate_decreasing_rate, self.min_learning_rate)
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
    
    def step_inference(self, x, y, action):
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
            return None
        return (x, y)

    def test(self, print_result=True, print_optimal_path=True):
        if (print_result):
            self.print_current_optimal_policy()
        reachedGoal = False
        x = self.environment.x_start
        y = self.environment.y_start
        path = []
        infinite_loop_counter = 0
        while (not reachedGoal and infinite_loop_counter < self.environment.x_size * self.environment.y_size):
            action = self.current_optimal_policy[x][y]
            path.append(action)
            x, y = self.step_inference(x, y, action)
            
            if ((x, y) == (self.environment.end_state.x, self.environment.end_state.y)):
                reachedGoal = True

            infinite_loop_counter += 1

        if print_optimal_path:
            print("OPTIMAL PATH:")    
            for action in path:
                print(str(action).split(".")[1], end=" ")

        return path

    # the two models have the same environment
    # this method can ONLY be used if the environments' grids look the same
    # simply path one of the models' environments to this method 
    @staticmethod
    def test_brute_force_combined_inference(model1, model2, environment, max_allowed_path_size=15, stack_max_capacity=1000, print_shortest_paths=True, k=2):
        paths = []
        shortest_paths = []
        stack = []
        shortest_paths_length = math.inf
        x, y = environment.x_start, environment.y_start
        start_state = environment.grid[x][y]
        stack.append((start_state, [], True))
        stack.append((start_state, [], False))

        while (len(stack) > 0) and (len(stack) < stack_max_capacity):
            current_state, current_path, use_policy1 = stack.pop()
            if (current_state == environment.end_state) and (current_path not in paths):
                paths.append(current_path)
                if (len(current_path) <= shortest_paths_length):
                    shortest_paths.append(current_path)
                    shortest_paths_length = len(current_path)
            else:
                if (use_policy1):
                    action = model1.current_optimal_policy[x][y]
                    x, y = model1.step_inference(current_state.x, current_state.y, action)
                    next_state = environment.grid[x][y]
                    next_path = current_path + [action]
                    if (len(current_path) < max_allowed_path_size):
                        stack.append((next_state, next_path, True))
                    if (len(current_path) < max_allowed_path_size):
                        stack.append((next_state, next_path, False))
                else:
                    action = model2.current_optimal_policy[x][y]
                    x, y = model2.step_inference(x, y, action)
                    next_state = environment.grid[x][y]
                    next_path = current_path + [action]
                    if (len(current_path) < max_allowed_path_size):
                        stack.append((next_state, next_path, True))
                    if (len(current_path) < max_allowed_path_size):
                        stack.append((next_state, next_path, False))
        if (print_shortest_paths):
            for i in range(len(shortest_paths)):
                print("Path No. " + str(i + 1) + ": ")
                for action in shortest_paths[i]:
                    print(str(action).split(".")[1], end=" ")
                print("\n")
        return paths, shortest_paths
    
    # the two models have the same environment
    # this method can ONLY be used if the environments' grids look the same
    # simply path one of the models' environments to this method 
    # choosing top k actionsi n each state for each policy
    @staticmethod
    def test_brute_force_combined_inference_2(model1, model2, environment, 
                                              max_allowed_path_size=15, stack_max_capacity=1000,
                                              print_shortest_paths=True, k=2):
        paths = []
        shortest_paths = []
        stack = []
        shortest_paths_length = math.inf
        x, y = environment.x_start, environment.y_start
        start_state = environment.grid[x][y]
        stack.append((start_state, []))

        while (len(stack) > 0) and (len(stack) < stack_max_capacity):
            current_state, current_path = stack.pop()
            if (current_state == environment.end_state) and (current_path not in paths):
                paths.append(current_path)
                if (len(current_path) <= shortest_paths_length):
                    shortest_paths_length = len(current_path)
            else:
                q_values_model1 = list(model1.environment.q_table[x][y])
                q_values_model2 = list(model2.environment.q_table[x][y])
                sorted_q_values_model1 = sorted(q_values_model1, reverse=True)
                sorted_q_values_model2 = sorted(q_values_model2, reverse=True)
                candidate_actions_model_1 = []
                candidate_actions_model_2 = []
                for i in range(min(k, len(environment.actions))):
                    max_action_index_model1 = q_values_model1.index(sorted_q_values_model1[i])
                    max_action_index_model2 = q_values_model2.index(sorted_q_values_model2[i])
                    action_model_1, action_model_2 = environment.actions[max_action_index_model1], environment.actions[max_action_index_model2]
                    candidate_actions_model_1.append(action_model_1)
                    candidate_actions_model_2.append(action_model_2)
                candidate_actions = list(set(candidate_actions_model_1 + candidate_actions_model_2))
                for action in candidate_actions:
                    x, y = model1.step_inference(current_state.x, current_state.y, action)
                    next_state = environment.grid[x][y]
                    next_path = current_path + [action]
                    if (len(current_path) < max_allowed_path_size):
                        stack.append((next_state, next_path))
            
        for path in paths:
            if (len(path) == shortest_paths_length):
                shortest_paths.append(path)
        
        if (print_shortest_paths):
            for i in range(len(shortest_paths)):
                print("Path No. " + str(i + 1) + ": ")
                for action in shortest_paths[i]:
                    print(str(action).split(".")[1], end=" ")
                print("\n")

        return paths, shortest_paths


    @staticmethod
    def check_contains_combined_path(paths, combined_path):
        for path in paths:
            if len(path) == len(combined_path):
                counter = 0
                for i in range(len(path)):
                    if (path[i] == combined_path[i]):
                        counter += 1
                if (counter == len(path)):
                    return True
        print("No equal path found...")
        return False                        








