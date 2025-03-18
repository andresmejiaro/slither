import numpy as np
import pygame
from tensorflow import keras
from preprocessing import add_closest
from tensorflow.keras import layers
import polars as pl
from preprocessing import add_closest_sl, preprocess_games





def softmax_rowise(X):
    exp_X = np.exp(X -np.max(X, axis=1, keepdims= True))
    return exp_X /np.sum(exp_X, axis=1, keepdims=True)

class Agent():
    def __init__(self):
        self.nn = keras.Sequential([
            layers.Dense(64, activation="relu", input_shape = (16,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(4, activation="linear")
        ])
        self.status = None

    def set_status(self, status):
        self.status = status

    def action_old(self):
        nkey = np.random.randint(0,4)
        keys = {"left":False,
                "right": False,
                "up": False,
                "down": False}
        match nkey:
            case 0:
                keys["up"] = True
            case 1:
                keys["down"] = True
            case 2:
                keys["right"] = True
            case 3:
                keys["left"] = True
            case _:
                pass
        return keys

    def action(self, df):
        df = add_closest_sl(df)
        inp = df[:,"north_view_W":"west_view_G"].to_numpy()
        Y = self.nn.predict(inp)
        rnum = np.random.random_sample()
        Y_soft = softmax_rowise(Y)
        Y_soft = np.cumsum(Y_soft)
        keys = {"left":False,
                "right": False,
                "up": False,
                "down": False}
        if (rnum < Y_soft[0]):
            keys["up"] =True
        elif (rnum < Y_soft[1]):
            keys["down"] =True
        elif (rnum < Y_soft[2]):
            keys["right"] =True
        else:
            keys["left"] =True
        return keys

    def model_update(self):
        if self.status is None:
            raise ValueError("Model doesn't have a valid status")
        self.nn.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse"  
        )
        df = preprocess_games(self.status.last_game)
        mapping = {"0": -1, "W": -100, "R": 10, "G": -10, "S": -100}
        df = df.filter(df["event"].is_not_null()).with_columns([
            pl.col("event").replace(mapping).cast(pl.Int32).alias("rewards")
        ])
        inp = df[:,"north_view_W":"west_view_G"].to_numpy()
        Y = self.nn.predict(inp)
        max_next = np.max(Y, axis = 1)
        action_categories = ["north", "south", "east", "west"]
        actions = df["direction"].to_numpy()
        indices = np.array([action_categories.index(a) for a in actions])
        onehot = np.eye(len(action_categories))[indices]
        alpha = 0.001
        max_next = np.tile(max_next[:,np.newaxis],4)
        r = max_next = np.tile(df["rewards"].to_numpy()[:,np.newaxis],4)
        gamma = 0.9
        Ynew = Y + alpha*( r  + gamma*max_next  - Y)*onehot
        self.nn.fit(inp,Ynew, epochs =1, batch_size = 1)
        



    
