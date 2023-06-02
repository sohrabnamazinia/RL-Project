import numpy as np
import math
import random
from Environment import Environment, StateType 
from Environment import Action 
from operator import itemgetter
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
from functools import reduce

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
        if (agent_state.x == self.environment.x_size - 1):
            Qs[1] = -math.inf
        if (agent_state.y == self.environment.y_size - 1):
            Qs[0] = -math.inf
        return Qs

    def select_action(self, agent_state):
        current_x = self.environment.agent_state.x
        current_y = self.environment.agent_state.y
        Qs = self.environment.q_table[current_x][current_y]
        action = self.get_best_action(Qs, agent_state)
        return action

    def get_best_action(self, Qs, agent_state):
        #Qs = self.apply_boundry_move_constraint(Qs, agent_state)
        index, element = max(enumerate(Qs), key=itemgetter(1))
        #print("best action", index, element)
        if index == 0:
            return Action.RIGHT
        elif index == 1:
            return Action.DOWN
        else:
            print("Action not defined")
            return None  
    
    def select_random_action(self, agent_state):
        possible_actions = self.environment.actions.copy()
        #print("agent state random", (agent_state.x, agent_state.y))
        if (agent_state.x == self.environment.x_size - 1):
            possible_actions.remove(self.environment.actions[1])
        if (agent_state.y == self.environment.y_size - 1):
            possible_actions.remove(self.environment.actions[0])
        if len(possible_actions) == 0:
            #print("No Possible Action!")
            return None
        else:
            #print("possible random", possible_actions)
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
                
                if action == None:
                    break

                old_state = self.environment.agent_state
                # update visited_power_states
                if (old_state.type == StateType.POWER):
                    visited_power_states.append(old_state)

                next_state, reward = self.environment.step(self.environment.agent_state, action, visited_power_states)
                if next_state == None:
                    break
                self.environment.q_table[old_state.x][old_state.y][action.value] = (1 - self.learning_rate) * self.environment.q_table[old_state.x][old_state.y][action.value] + self.learning_rate * (reward + self.environment.discount_factor * (max(self.environment.q_table[next_state.x][next_state.y])))
                total_episode_reward += reward
                if (self.environment.reached_goal()):
                    break
                # exploration-exploitation tradeoff: linear decay implementaiton
                self.learning_rate = max(self.learning_rate - self.learning_rate_decreasing_rate, self.min_learning_rate)
                self.exploration_prob = max(self.exploration_prob - self.exploration_decreasing_rate, self.min_exploration_prob)
            self.rewards_per_episode.append(total_episode_reward)
        return self.get_current_optimal_policy(), visited_power_states

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
        if (action == Action.RIGHT):
            x, y = x, min(y + 1, self.environment.y_size - 1)
        elif (action == Action.DOWN):
            x, y = min(x + 1, self.environment.x_size - 1), y
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
    # choosing top k actions in each state for each policy
    @staticmethod
    def test_brute_force_combined_inference_2(model1, model2, environment, 
                                              max_allowed_path_size=15, stack_max_capacity=100000,
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
            x, y = current_state.x, current_state.y
            if (current_state == environment.end_state):
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
                    print("action", action)
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
    def test_brute_force_combined_inference_3(model1, model2, environment,
                                              max_allowed_path_size=15, stack_max_capacity=100000,
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
            x, y = current_state.x, current_state.y
            if (current_state == environment.end_state):
                paths.append(current_path)
                # print("yes", current_path)
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
                    action_model_1, action_model_2 = environment.actions[max_action_index_model1], environment.actions[
                        max_action_index_model2]
                    candidate_actions_model_1.append(action_model_1)
                    candidate_actions_model_2.append(action_model_2)
                candidate_actions = list(set(candidate_actions_model_1 + candidate_actions_model_2))
                for action in candidate_actions:
                    x, y = model1.step_inference(current_state.x, current_state.y, action)
                    next_state = environment.grid[x][y]
                    next_path = current_path + [((current_state.x, current_state.y), (next_state.x, next_state.y))]
                    if (len(current_path) < max_allowed_path_size):
                        stack.append((next_state, next_path))

        for path in paths:
            if (len(path) == shortest_paths_length and path not in shortest_paths):
                shortest_paths.append(path)

        if (print_shortest_paths):
           for i in range(len(shortest_paths)):
              print("Path No. " + str(i + 1) + ": ")
              for (s1, s2) in shortest_paths[i]:
                  print(str(s1) + "->" + str(s2), end=" || ")
              print("\n")

        return paths, shortest_paths

    @staticmethod
    def getRemovableEdges(G, edgeLst, initNodes):

        removableEdgeLst = []
        for (u, v) in edgeLst:
            G.remove_edge(u, v)
            f = nx.floyd_warshall(G)
            print("f", f)
            addEdge = True
            for s in initNodes:
                if 'inf' in list(map(str, f[s].values())):
                    G.add_edge(u, v)
                    addEdge = False
                    break
            if addEdge:
                removableEdgeLst.append((u, v))
        return removableEdgeLst

    @staticmethod

    def buildDAG(env, model, lst):

        edges = set()
        for i in range(len(lst)):
            path = lst[i]
            for e in path:
                if e[0] != e[1] and e[1] != (env.x_start, env.y_start):
                    edges.add(e)

        edges = list(edges)
        G = nx.DiGraph()
        G.add_edges_from(edges)
        # print("before", len(G.edges()))
        # if not nx.is_directed_acyclic_graph(G):
        #     eL = model.getRemovableEdges(G, list(G.edges()), [(env.x_start, env.y_start), (env.end_state.x, env.end_state.y)])
        #     print("list", eL)
        #     G.remove_edges_from(eL)
        # print("after", len(G.edges()))
            # print("Cycle", list(nx.simple_cycles(G)))
            # cycle = list(nx.simple_cycles(G))
            # for i, c in enumerate(cycle):
            #     print("Cycle " + str(i) +":", c)
            # edge = random.choice(cycle)
            # G.remove_edge(edge[0], edge[1])
            # print("Cycle", nx.find_cycle(G))
        print("Is Graph a DAG?", nx.is_directed_acyclic_graph(G))

        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size=500, node_color="lightblue")
        nx.draw_networkx_edges(G, pos, edge_color='b', edge_cmap=plt.cm.Blues, arrows=True, width=2.5,  arrowstyle="->", arrowsize=15)
        nx.draw_networkx_labels(G, pos)

        plt.show()

        return G

    @staticmethod
    def backtrack(G, env, model, visited_power_states):

        boundry = {node: [[0, 0]] * env.action_size for node in G.nodes}

        inDeg = {n: [] for n in G.nodes()}
        outDeg = {n: [] for n in G.nodes()}

        for e in G.edges():
            node1 = e[0]
            node2 = e[1]
            outDeg[node1].append(node2)
            inDeg[node2].append(node1)

        q = deque([(env.end_state.x, env.end_state.y)])
        visited = set()

        while q:
            node = q.popleft()
            visited.add(node)
            for i in inDeg[node]:
                print("No", i, node)
                # print("node", node, i, inDeg[i])

                next_state = env.grid[node[0]][node[1]]
                prev_state = env.grid[i[0]][i[1]]

                _, action = model.obtainAction(prev_state, next_state)

                # print("hello", action, boundry[node])

                reward = env.compute_reward(prev_state, next_state, env.policy, visited_power_states)
                lst = boundry[node]
                LB = model.learning_rate * (reward)
                UB = pow(-1,  (model.episode_count - 1)) * reward * (
                    pow((1 - model.learning_rate), (model.episode_count)) + pow(-1, (
                                model.episode_count - 1))) + model.learning_rate * (
                                 env.discount_factor * (max(reduce(lambda x,y :x+y ,lst))))

                # update visited_power_states
                if (prev_state.type == StateType.POWER) and prev_state in visited_power_states:
                    visited_power_states.remove(prev_state)

                boundry[i][action] = [round(LB, 2), round(UB, 2)]
                if i not in q and i not in visited:
                    q.append(i)


        return boundry, outDeg

    @staticmethod
    def pruning(G, model, env, adjList, boundry):

        q = deque()
        q.append((env.x_start, env.y_start))
        visited = set()
        while q:
            node = q.popleft()
            visited.add(node)
            prev_state = env.grid[node[0]][node[1]]
            remove = []
            # print("node", node, boundry[node], adjList[node])
            if len(adjList[node]) == 1:
                q.append(adjList[node][0])
            else:
                for i, n in enumerate(adjList[node]):
                    next_state = env.grid[n[0]][n[1]]
                    _, action_n = model.obtainAction(prev_state, next_state)
                    lB = boundry[node][action_n][0]
                    uB = boundry[node][action_n][1]
                    # print("boundry n", n, action_n, lB, uB)
                    for m in adjList[node]:

                        if n in remove or m in remove or n == m:
                            continue
                        else:
                            nextState = env.grid[m[0]][m[1]]
                            _, action_m = model.obtainAction(prev_state, nextState)
                            bound = boundry[node][action_m]
                            # print("boundry m", m, action_m, bound[0], bound[1])
                            if bound[1] <= lB:
                                remove.append((node, m))
                            else:
                                if m not in q and m not in visited:
                                    q.append(m)
                if remove != []:
                    print("lists", remove)

                G.remove_edges_from(remove)

        return G

    @staticmethod
    def findPath(env, G):
        paths = []
        for path in nx.all_simple_paths(G, source= (env.x_start, env.y_start), target = ((env.end_state.x, env.end_state.y))):
            paths.append(path)
        return paths

    @staticmethod
    def obtainAction(state1, state2):

        x1, y1 = state1.x, state1.y
        x2, y2 = state2.x, state2.y

        action = None
        index = None

        if x1 == x2 and y1 + 1 == y2:
            action = "right"
            index = 0
        elif x1 + 1 == x2 and y1 == y2:
            action = "down"
            index = 1

        return action, index

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








