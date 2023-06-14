from Model import Model
from Environment import Environment
from Environment import Policy
import matplotlib.pyplot as plt
import utilities


env1 = Environment(policy=Policy.CLOSENESS)
env2 = Environment(policy=Policy.MAXPOWER)
env3 = Environment(policy=Policy.COMBINATION)
model1 = Model(env1, episode_count=1000)
model2 = Model(env2, episode_count=1000)
model3 = Model(env3, episode_count=1000)
model1.environment.print_grid()
train_result_1, visited_power_states_1 = model1.train()
train_result_2, visited_power_states_2 = model2.train()
train_result_3, visited_power_states_3 = model3.train()
print("Closeness policy:")
model1.test()
print("\n*****************************\n")
print("Maxpower policy:")
model2.test()
print("\n*****************************\n")
print("Combined Policy:")
model3_optimal_path = model3.test()
ground_truth_cumulative_reward = model3.environment.compute_reward_ground_truth_path(model3_optimal_path)
print("\nGround Truth Cumulative Reward = " + str(ground_truth_cumulative_reward))
print("\n******************************\n")
print("Brute Force for finding the combined policy")
brute_force_paths, shortest_paths = Model.test_brute_force_combined_inference_3(model1, model2, env1, k=2, print_shortest_paths=False, max_allowed_path_size=model3.max_iter_per_episode)
print("\n******************************\n")
graph = Model.buildDAG(env3, model3, brute_force_paths)
boundry, adjList = Model.backtrack(graph, env3, model3, visited_power_states_3)
G = Model.pruning(graph, model3, env3, adjList, boundry)
paths = Model.findPath(env3, G)
utilities.calculate_pruning_percentage(brute_force_paths, paths)
rewards = model3.environment.compute_reward_dag_paths(paths, ground_truth_cumulative_reward)
print("\n******************************\n")
rewards_difference = [ground_truth_cumulative_reward - reward for reward in rewards]
utilities.plot_path_reward_defference(rewards_difference)
print("\n******************************\n")



# NOTE: For now, we do not need this method because we believe our brute force method
#       works properly. Remember that this method does not work with brute_force_version_3
#       because the elements of the returning paths of this method are not ACTIONS, but tuples.
# print("Does the brute force resulting path list contains the combined policy resulting path?")
# if (Model.check_contains_combined_path(paths, model3_optimal_path)):
#     print("\tYes")
# else:
#     print("\tNo")