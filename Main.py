from Model import Model
from Environment import Environment

env = Environment()
model = Model(env)
result = model.train()
print(result)