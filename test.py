#%%
import polars as pl
import numpy as np
import keras

##

from src.Agent import Agent
from src.preprocessing import data_augmentation, add_closest
# %%

miniagent = Agent(load="models/learn3.keras", save ="models/curated.keras")


# %%

db = pl.read_csv("logs/10000_random3.csv")

#%%

#valid_ids = db.filter(pl.col("reward") > 0).select("game_id").unique()
#filtered = db.join(valid_ids, on="game_id")

#%%

#miniagent.replay_train(filtered)


# %%


#miniagent.nn.save("models/learn1.keras")


#%%

# %%


#np.mean(Y,axis = 0)
#np.var(Y,axis = 0)
# %%
weights = miniagent.nn.layers[-1].get_weights()
print("Kernel:", weights[0])
print("Biases:", weights[1])
# %%


#%%

db2 = db.filter(pl.col("game_id")== "20250418_134623_4889")
db2["action"].tail()

#%%
inp, Y, update = miniagent.update_data_prep(db2)
# %%

update[-10:,]
# %%


#%%
import tensorflow as tf


#%%
loss_fn = keras.losses.get("mse")

#%%

# Ensure the inputs are tensors
#db = pl.read_csv("logs/10000_random.csv")
#db2 = db.filter(pl.col("game_id")== "20250417_153554_7766")



#%%
#inp, Y,  update = miniagent.update_data_prep(db2)
#%%
Y_pred = tf.convert_to_tensor(Y, dtype=tf.float32)
Y_target = tf.convert_to_tensor(Y +update, dtype=tf.float32)

# Call the exact loss function Keras uses
loss = loss_fn(Y_target, Y_pred)

print("Shape of loss:", loss.shape)
print("Per-sample loss:", loss.numpy())  # vector of losses
print("Mean loss (as shown in Keras logs):", tf.reduce_mean(loss).numpy())

# %%

miniagent.nn.fit(inp, Y + update)
# %%
