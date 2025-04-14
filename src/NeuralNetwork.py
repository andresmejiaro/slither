#%%
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

#%%
closest_model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape = (16,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(4, activation="linear")
])
# %%




closest_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="mse"  
)
# %%


inp = df.filter(pl.col("event").is_not_null())[:,"north_view_W":"west_view_G"].to_numpy()

# %%

Y =closest_model.predict(inp)
# %%
Y
# %%
# Processing output
mapping = {"0": -1, "W": -100, "R": 10, "G": -10, "S": -100}

df = df.filter(df["event"].is_not_null()).with_columns([
    pl.col("event").replace(mapping).cast(pl.Int32).alias("rewards")
])


# %%

max_next = np.max(Y, axis = 1)
# %%


action_categories = ["north", "south", "east", "west"]

actions = df["direction"].to_numpy()
# %%

indices = np.array([action_categories.index(a) for a in actions])

# %%
onehot = np.eye(len(action_categories))[indices]
# %%

alpha = 0.001

max_next = np.tile(max_next[:,np.newaxis],4)

r = max_next = np.tile(df["rewards"].to_numpy()[:,np.newaxis],4)

gamma = 0.9

# %%

Ynew = Y + alpha*( r  + gamma*max_next  - Y)*onehot
# %%


closest_model.fit(inp,Ynew, epochs =1, batch_size = 1)
# %%
