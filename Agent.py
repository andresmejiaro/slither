import numpy as np
import pygame
import polars as pl
import os
import sys
from tensorflow import keras
from tensorflow.keras import initializers
from tensorflow.keras.models import load_model
from preprocessing import add_closest, load_filter_data
from tensorflow.keras import layers
from preprocessing import add_closest_sl, preprocess_games, data_augmentation
from game_settings import rewards, gamma, alpha, episode_passes, temperature, close_reward_multiplier
np.set_printoptions(precision=2, suppress=True)



def softmax_rowise(X):
    exp_X = np.exp(X -np.max(X, axis=1, keepdims= True))
    return exp_X /np.sum(exp_X, axis=1, keepdims=True)

class Agent():
    def __init__(self,save = None, load = None):
        self.save = save
        self.load = load
        
        
        if self.load is None or not os.path.exists(self.load):
            self.nn = keras.Sequential([
                layers.Dense(16, input_shape = (16,)),
                layers.LeakyReLU(negative_slope= 0.1),
                layers.Dense(32),
                layers.LeakyReLU(negative_slope= 0.1),
                layers.Dense(16),
                layers.LeakyReLU(negative_slope= 0.1),
                layers.Dense(4, activation="linear",bias_initializer=initializers.constant(rewards["W"]))
            ])
        else:
            try:
                self.nn = load_model(self.load)
            except:
                print("Fatal! Cannot load model (make sure is .h5)")
                sys.exit(1)
        self.nn.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss="mse"  
        )
        self.status = None
        self.temperature = temperature
        self.game_history = pl.DataFrame()
        self.epsilon = 0

    def set_status(self, status):
        self.status = status


    def action(self, df):
        df = add_closest_sl(df)
        inp = df[:,"north_view_W":"west_view_G"].to_numpy()
        Y = self.nn.predict(inp, verbose = 0)
        print(f"Yor: {Y}")
        Y = Y/self.temperature
        rnum = np.random.random_sample()
        Y_soft = softmax_rowise(Y)
        print("UP DOWN RiGHT LEFT")
        print(f"Y: {Y}")
        print(f"Y prob: {Y_soft}")
        ep2 = min(self.epsilon + 0.01   , 1)
        Y_prob_end = ep2*np.array([1,1,1,1])/4  + ((1-ep2))*Y_soft
        print(f"Y prob end: {Y_prob_end}")
        Y_soft = np.cumsum(Y_prob_end)
        keys = {"left":False,
                "right": False,
                "up": False,
                "down": False}
        if (rnum < Y_soft[0]):
            keys["up"] =True
            print("UP")
        elif (rnum < Y_soft[1]):
            keys["down"] =True
            print("DOWN")
        elif (rnum < Y_soft[2]):
            keys["right"] =True
            print("RIGHT")
        else:
            keys["left"] =True
            print("LEFT")
        return keys
    #Remember 0 is up
    #1 is down
    #2 is right
    #3 is left
    
    def update_data_prep(self, df = None):
        if df is None:
            df = data_augmentation(self.status.last_game)
        else:
            data_augmentation(df)
        df = preprocess_games(df)
        df = df.filter(df["event"].is_not_null()).with_columns([
            pl.col("event").replace(rewards).cast(pl.Int32).alias("rewards")
        ])
        inp = df[:,"north_view_W":"west_view_G"].to_numpy()
        close_reward = close_reward_multiplier * distance_reward(df)
        Y = self.nn.predict(inp)
        max_next = np.max(Y, axis = 1)
        action_categories = ["north", "south", "east", "west"]
        actions = df["direction"].to_numpy()
        indices = np.array([action_categories.index(a) for a in actions])
        onehot = np.eye(len(action_categories))[indices]
        max_next = np.tile(max_next[:,np.newaxis],4)
        weights =np.log(np.abs(df["rewards"].to_numpy() + close_reward) +1 )
        r = np.tile((df["rewards"].to_numpy() + close_reward)[:,np.newaxis],4)
        update = alpha*( r  + gamma*max_next  - Y)*onehot
        return (inp, Y, update, weights)
    
    def model_update(self):
        if self.status is None:
            raise ValueError("Model doesn't have a valid status")
        inp, Y, update, weights = self.update_data_prep()
        Ynew = Y + update
        self.nn.fit(inp,Ynew, epochs = episode_passes, batch_size = 1) #, sample_weight = weights
    
    def replay_train(self):
        df = load_filter_data()
        inp, Y, update, weights = self.update_data_prep(df)
        Ynew = Y + update
        self.nn.fit(inp,Ynew, epochs = episode_passes, batch_size = 256)

def distance_reward(df):
    distances = df.select(pl.selectors.ends_with("_G")).to_numpy()
    masked_distances = np.where(distances == -1, np.inf, distances)
    row_min = np.min(masked_distances, axis=1)
    return np.maximum(10-row_min,0)