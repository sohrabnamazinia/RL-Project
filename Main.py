from Model import Model
from Environment import Environment
from Environment import Policy

env1 = Environment(policy=Policy.CLOSENESS)
env2 = Environment(policy=Policy.MAXPOWER)
env3 = Environment(policy=Policy.COMBINATION)
model1 = Model(env1, episode_count=10000)
model2 = Model(env2, episode_count=10000)
model3 = Model(env3, episode_count=10000)
model1.environment.print_grid()
train_result_1 = model1.train()
train_result_2 = model2.train()
train_result_3 = model3.train()
print("Closeness policy:")
model1.test()
print("\n*****************************\n")
print("Maxpower policy:")
model2.test()
print("\n*****************************\n")
print("Brute Force for finding the combined policy:")
paths, shortest_paths = Model.test_brute_force_combined_inference_2(model1, model2, env1, k=1)
print("\n******************************\n")
print("Combined Policy:")
model3_optimal_path = model3.test()
print("\n******************************\n")
print("Does the brute force resulting path list contains the combined policy resulting path?")
if (Model.check_contains_combined_path(paths, model3_optimal_path)):
    print("\tYes")
else:
    print("\tNo")