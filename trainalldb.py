#%%
import polars as pl
import numpy as np
import keras

##

from src.Agent import Agent
from src.preprocessing import data_augmentation, add_closest

from pathlib import Path
# %%

log_files = sorted(Path("logs").glob("*.csv"))



#%%

modelname = "models/128-64-32-16-4-regular-exp.keras"

miniagent = Agent(load=modelname, save =modelname)


# %%


#%%


for i in range(2):
     for fn in log_files:
        print(f"{i} {fn}")
        db = pl.read_csv(fn) 
        miniagent.replay_train(db)


# %%


miniagent.nn.save(filepath=modelname)


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
