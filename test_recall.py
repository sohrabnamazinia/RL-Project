# environments are all squares in this experiment
# environment sizes increase by a multiplication of 2
# input is the diff_env_count and diff_start_count

import random
import utilities
from Model import Model
from Environment import Environment
from Environment import Policy

def run_one_test(start_x, start_y, env_side_length):
    env1 = Environment(policy=Policy.CLOSENESS, x_size=start_x, y_start=start_y, hard_reset_index=4)
    env2 = Environment(policy=Policy.MAXPOWER, x_size=start_x, y_start=start_y, hard_reset_index=4)
    env3 = Environment(policy=Policy.COMBINATION, x_size=start_x, y_start=start_y, hard_reset_index=4)
    max_iter_per_episode = (env_side_length - 1) * 2 + 2
    episode_count = 1000 * (env_side_length / 5)
    model1 = Model(env1, episode_count=episode_count, max_iter_per_episode=max_iter_per_episode)
    model2 = Model(env2, episode_count=episode_count, max_iter_per_episode=max_iter_per_episode)
    model3 = Model(env3, episode_count=episode_count, max_iter_per_episode=max_iter_per_episode)
    print("Print grid - recall test for start_x = %d, start_y = %d, env_side_length = %d", start_x, start_y, env_side_length)
    model1.environment.print_grid()
    train_result_1, visited_power_states_1 = model1.train()
    train_result_2, visited_power_states_2 = model2.train()
    train_result_3, visited_power_states_3 = model3.train()
    print("Closeness policy - recall test for start_x = %d, start_y = %d, env_side_length = %d", start_x, start_y, env_side_length)
    model1.test()
    print("\n*****************************\n")
    print("MaxPower policy - recall test for start_x = %d, start_y = %d, env_side_length = %d", start_x, start_y, env_side_length)
    model2.test()
    print("\n*****************************\n")
    print("Combined policy - recall test for start_x = %d, start_y = %d, env_side_length = %d", start_x, start_y, env_side_length)
    model3_optimal_path = model3.test()
    print("\n******************************\n")
    print("For recall test for start_x = %d, start_y = %d, env_side_length = %d", start_x, start_y, env_side_length + ", Brute Force for finding the combined policy:\n")
    brute_force_paths, shortest_paths = Model.test_brute_force_combined_inference_3(model1, model2, env1, k=2, print_shortest_paths=False, max_allowed_path_size=model3.max_iter_per_episode)
    print("\n******************************\n")
    max_reward_brute_force_path = env3.get_max_reward_brute_force_path(brute_force_paths)
    graph = Model.buildDAG(env3, model3, brute_force_paths)
    boundry, adjList = Model.backtrack(graph, env3, model3, visited_power_states_3)
    G = Model.pruning(graph, model3, env3, adjList, boundry)
    paths = Model.findPath(env3, G)
    result = utilities.check_path_in_paths(max_reward_brute_force_path, paths)
    print("\n******************************\n")
    return result


diff_env_count = 2
diff_start_count = 2
environment_size_lengths = []
for i in range(diff_env_count):
    environment_size_lengths.append((i + 1) * 5)

recalls = []
for i in range(diff_env_count):
    env_side_length = environment_size_lengths[i]
    start_positions = []
    true_count = 0
    for j in range(diff_start_count):
        start_x = random.randint(0, env_side_length - 1)
        start_y = random.randint(0, env_side_length - 1)
        start_positions.append((start_x, start_y))
    for (start_x, start_y) in start_positions:
        true_count += run_one_test(start_x, start_y, env_side_length)
    recalls.append((true_count / diff_start_count) * 100)
