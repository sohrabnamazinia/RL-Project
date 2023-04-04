from Model import Model
from Environment import Environment
from Environment import Policy

env = Environment(policy=Policy.CLOSENESS)
model = Model(env)
model.environment.print_grid()
train_result = model.train()
model.test(print_result=True)
#print(model.rewards_per_episode)