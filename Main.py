from Model import Model
from Environment import Environment

env = Environment()
model = Model(env)
model.environment.print_grid()
train_result = model.train()
model.test(print_result=True)
