#%%
import polars as pl
import numpy as np
import keras

##

from src.Agent import Agent
from src.preprocessing import data_augmentation, add_closest
# %%

miniagent = Agent(load="models/anew2.keras", save ="models/anew2.keras")


# %%

db = pl.read_csv("logs/10000_random.csv")

#%%

valid_ids = db.filter(pl.col("reward") > 0).select("game_id").unique()
filtered = db.join(valid_ids, on="game_id")

#%%

miniagent.replay_train_individual(filtered)


# %%


miniagent.nn.save(filepath="models/anew2.keras")


#%%

# %%


#np.mean(Y,axis = 0)
#np.var(Y,axis = 0)
# %%
#weights = miniagent.nn.layers[-1].get_weights()
#print("Kernel:", weights[0])
#print("Biases:", weights[1])
# %%


#%%
