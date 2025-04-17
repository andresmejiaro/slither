#%%
import polars as pl

##

from src.Agent import Agent
from src.preprocessing import data_augmentation
# %%

miniagent = Agent()


# %%


db = pl.read_csv("snakelogs.csv")
# %%

one_game = db.filter(pl.col("game_id") == "20250415_192649")

# %%

eight_games = data_augmentation(one_game)
# %%
