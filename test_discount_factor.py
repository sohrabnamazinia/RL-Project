from Model import Model
from Environment import Environment
from Environment import Policy
import matplotlib.pyplot as plt
import utilities

# This variable is the input of this file
experiments_count = 10
# This variable is the output of this file
pruning_percentages = []

def set_discount_factors(n):
    discount_factors = [0]
    for i in range(1, n):
        discount_factors.append(discount_factors[i - 1] + (1 / experiments_count))
    return discount_factors

discount_factors = set_discount_factors(experiments_count)

for i in range(experiments_count):
    d_f = discount_factors[i]
    env1 = Environment(policy=Policy.CLOSENESS, discount_factor=d_f, x_size=100, y_size=100, hard_reset_index=3)
    env2 = Environment(policy=Policy.MAXPOWER, discount_factor=d_f, x_size=100, y_size=100, hard_reset_index=3)
    env3 = Environment(policy=Policy.COMBINATION, discount_factor=d_f, x_size=100, y_size=100, hard_reset_index=3)
    model1 = Model(env1, episode_count=10000, max_iter_per_episode=200)
    model2 = Model(env2, episode_count=10000, max_iter_per_episode=200)
    model3 = Model(env3, episode_count=10000, max_iter_per_episode=200)
    print("Environment for discount_facor = " + str(d_f))
    model1.environment.print_grid()
    train_result_1, visited_power_states_1 = model1.train()
    train_result_2, visited_power_states_2 = model2.train()
    train_result_3, visited_power_states_3 = model3.train()
    print("Closeness policy for discount_factor = " + str(d_f))
    model1.test()
    print("\n*****************************\n")
    print("Maxpower policy for discpunt_factor = " + str(d_f))
    model2.test()
    print("\n*****************************\n")
    print("Combined Policy for discount_factor = " + str(d_f))
    model3_optimal_path = model3.test()

    #ground_truth_cumulative_reward = model3.environment.compute_reward_ground_truth_path(model3_optimal_path)
    #rint("\nGround Truth Cumulative Reward for discount_factor = " + str(d_f) + " is " + str(ground_truth_cumulative_reward))
    print("\n******************************\n")
    print("Brute Force for finding the combined policy for discount_factor = " + str(d_f))
    brute_force_paths, shortest_paths = Model.test_brute_force_combined_inference_3(model1, model2, env1, k=2, print_shortest_paths=False, max_allowed_path_size=model3.max_iter_per_episode)
    print("\n******************************\n")
    graph = Model.buildDAG(env3, model3, brute_force_paths, plot_dag=False)
    boundry, adjList = Model.backtrack(graph, env3, model3, visited_power_states_3)
    G = Model.pruning(graph, model3, env3, adjList, boundry)
    paths = Model.findPath(env3, G)
    print("Calculating pruning percentage for discount_factor = " + str(d_f))
    result = utilities.calculate_pruning_percentage(brute_force_paths, paths)
    pruning_percentages.append(result)
    #rewards = model3.environment.compute_reward_dag_paths(paths, ground_truth_cumulative_reward)
    #print("\n******************************\n")
    #rewards_difference = [ground_truth_cumulative_reward - reward for reward in rewards]
    #utilities.plot_path_reward_defference(rewards_difference)
    print("\n******************************\n")

utilities.plot_discount_factors(discount_factors, pruning_percentages) 
print("Experiment results:\n")
print("Discount factors: " + str(discount_factors))
print("Pruning Percentages: " + str(pruning_percentages))
print("*****************************")