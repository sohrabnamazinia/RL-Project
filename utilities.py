import matplotlib.pyplot as plt
import numpy as np

def calculate_pruning_percentage(paths, paths_pruned):
    a = len(paths)
    b = len(paths_pruned)
    result = ((a - b) / a) * 100
    print("\nPruning Percentage: %" + str(result) + "\n")
    return result

def plot_path_reward_defference(rewards_difference):
    plt.plot(rewards_difference)
    plt.xlabel('Path Index')
    plt.ylabel('Ground truth reward - DAG Paths Rewards')
    plt.title('Rewards Difference')
    plt.show()
    print("Ground truth reward - DAG Paths Rewards has been plotted")

def plot_discount_factors(discount_factors, pruning_percentages):
    x = np.array(discount_factors)
    y = np.array(pruning_percentages)
    plt.xlabel("Discount Factor")
    plt.ylabel("Pruning Percentage")
    plt.title("Pruning Percentage Experiment")
    plt.xticks(x)
    plt.plot(x, y)
    plt.show()

def check_path_in_paths(key_path, paths):
    for path in paths:
        count = 0
        if len(key_path) + 1 != len(path):
            continue
        for i in range(len(key_path)):
            if (key_path[i][0] == path[i]):
                count += 1
        if (key_path[len(path) - 2][1] == path[len(path) - 1]):
                count += 1
        if count == len(path):
            return True
    return False

def plot_recalls(env_areas, recalls):
    x = np.array(env_areas)
    y = np.array(recalls)
    plt.xlabel("Environment side length")
    plt.ylabel("Recall Percentage")
    plt.title("Recall Percentage Experiment")
    plt.xticks(x)
    plt.plot(x, y)
    plt.show()