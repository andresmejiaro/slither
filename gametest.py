#%%
from src.Game import Game
from src.Interpreter import Interpreter
from src.Agent import Agent
import numpy as np
#%%

interpreter = Interpreter()
agent = Agent(input_size=66, output_size=4, alpha= 0.1,gamma=0.9, nn_learningrate=0.001)

#%%
# %%



for i in range (100):
    game = Game()
    Interpreter.agent_loop(agent,game,epsilon=1)
    game.export_episode(episode_id=f"ep{i}")
# %%
