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