import numpy as np
import pygame
import polars as pl
import os
import sys
from tensorflow import keras
from tensorflow.keras import initializers, layers
from tensorflow.keras.models import load_model
from . import add_closest, load_filter_data, add_closest_sl, preprocess_games, data_augmentation, add_reward, align_status_rewards, direction_one_hot
from game_settings import rewards, gamma, alpha, episode_passes, close_reward_multiplier, nn_learningrate
np.set_printoptions(precision=2, suppress=True)
from tqdm import tqdm


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
                layers.Dense(64),
                layers.LeakyReLU(negative_slope= 0.1),
                layers.Dense(32),
                layers.LeakyReLU(negative_slope= 0.1),
                layers.Dense(16),
                layers.LeakyReLU(negative_slope= 0.1),
                #layers.Dense(4, activation="linear",bias_initializer=initializers.constant(rewards["W"]))
                layers.Dense(4, activation="linear")
            ])
        else:
            try:
                self.nn = load_model(self.load)
            except:
                print("Fatal! Cannot load model (make sure is .h5 or .keras)")
                sys.exit(1)
        self.nn.compile(
            optimizer=keras.optimizers.Adam(learning_rate=nn_learningrate),
            loss="mse"  
        )
        self.game_history = pl.DataFrame()
        self.epsilon = 0
        self.status = None

    def set_status(self, status):
        self.status = status

    def model_update(self):
        if self.status is None:
            raise ValueError("Model doesn't have a valid status")
        print(self.status.last_game)
        df = data_augmentation(self.status.last_game)
        games = df["game_id"].unique()
        for game in games:
            df2 = df.filter(pl.col("game_id") == game)
            inp, Y, update = self.update_data_prep(df2)
            Ynew = Y + update
            self.nn.fit(inp,Ynew, epochs = episode_passes, batch_size = 1) #, sample_weight = weights


    def action(self, df):
        df = add_closest_sl(df)
        inp = df[:,"north_view_W":"west_view_G"].to_numpy()
        Y = self.nn.predict(inp, verbose = 0)
        rnum = np.random.random_sample()
        Y_soft = softmax_rowise(Y)
        print("NORTH SOUTH EAST WEST")
        print(f"Y: {Y}")
        print(f"Y prob: {Y_soft}")
        Y_prob_end = self.epsilon*np.array([1,1,1,1])/4  + ((1-self.epsilon))*Y_soft
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




    def update_data_prep(self, df):
        df = preprocess_games(df)
        inp = df[:,"north_view_W":"west_view_G"].to_numpy()
        close_reward = close_reward_multiplier * distance_reward(df)
        Y = self.nn.predict(inp)
        max_next = np.max(Y, axis = 1)
        game_ouputs = pl.DataFrame({"game_id":df["game_id"],"max_next":max_next})
        game_ouputs = game_ouputs.with_columns(
            pl.col("max_next").shift(-1).over("game_id").fill_null(0)
        )
        max_next = game_ouputs[:,"max_next"].to_numpy()
        max_next = np.tile(max_next[:,np.newaxis],4)
        onehot = direction_one_hot(df)
        #weights =np.log(np.abs(df["reward"].to_numpy() + close_reward) +1 )
        r = np.tile((df["reward"].to_numpy() + close_reward)[:,np.newaxis],4)
        update = alpha*( r  + gamma*max_next  - Y)*onehot
        return (inp, Y, update)
    
    def replay_train(self, df):
        df = data_augmentation(df)
        inp, Y, update = self.update_data_prep(df)
        Ynew = Y + update
        self.nn.fit(inp,Ynew, epochs = episode_passes, batch_size = 256)
    

    def replay_train_individual(self, df):
        df = data_augmentation(df)
        games = df["game_id"].unique()
        for game in tqdm(games, descr ="Training on replay"):
            df2 = df.filter(pl.col("game_id") == game)
            inp, Y, update = self.update_data_prep(df2)
            Ynew = Y + update
            self.nn.fit(inp,Ynew, epochs = episode_passes, batch_size = 1) #, sample_weight = weights
        

def distance_reward(df):
    distances = df.select(pl.selectors.ends_with("_G")).to_numpy()
    masked_distances = np.where(distances == -1, np.inf, distances)
    row_min = np.min(masked_distances, axis=1)
    return np.maximum(10-row_min,0)