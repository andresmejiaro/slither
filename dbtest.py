#%%

from src.Interpreter import Interpreter
from src.Game import Game
from game_settings import rewards
from src.Agent import Agent
import pygame
from game_settings import screen_width, screen_height

#%%


base = Interpreter.load_log_file("logs/games.jsonl")
# %%

base = Interpreter.rewards_to_numeric(base,rewards)
base


#%%


agente = Agent(input_size=66, output_size=4, alpha=0.1, gamma=0.9, nn_learningrate=0.001)

#%%

agente.replay_train(base)
# %%

#agente.replay_train_individual(base)
# %%

pygame.init()
screen =screen = pygame.display.set_mode((screen_width, screen_height))

for i in range(20):
    gam = Game(screen=screen)
    Interpreter.agent_loop(agente,gam, visual=True)

pygame.quit()
# %%
