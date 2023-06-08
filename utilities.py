def calculate_pruning_percentage(paths, paths_pruned):
    a = len(paths)
    b = len(paths_pruned)
    result = ((a - b) / a) * 100
    print("\nPruning Percentage: %" + str(result) + "\n")
    return result