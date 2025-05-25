
import os
from tqdm import tqdm
import numpy as np
import polars as pl
import sys
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from game_settings import gamma, alpha, nn_learningrate
np.set_printoptions(precision=2, suppress=True)


class Agent():
    def __init__(
            self,
            save=None,
            load=None,
            input_size=1,
            output_size=1,
            l2=0.001,
            alpha=alpha,
            gamma=gamma,
            nn_learningrate=nn_learningrate):
        """
        if given load and save parameters it will try to read the model and
        save to the given files. If the file can't be opened it creates a
        network with fixed topology except for the input_size and output_size.
        If for any reason you want a custom topology create the neural network
        and load it.
        Output size is relevant for the action chooser. You should also give it
        even if the topology is given
        """
        self.savef = save
        self.load = load
        self.alpha = alpha
        self.gamma = gamma
        self.nn_learningrate = nn_learningrate
        self.output_size = output_size
        if self.load is None or not os.path.exists(self.load):

            self.nn = keras.Sequential([
                layers.Dense(128, input_shape=(input_size,),
                             kernel_regularizer=regularizers.l2(l2)),
                layers.LeakyReLU(negative_slope=0.1),
                layers.Dense(64, kernel_regularizer=regularizers.l2(l2)),
                layers.LeakyReLU(negative_slope=0.1),
                layers.Dense(32, kernel_regularizer=regularizers.l2(l2)),
                layers.LeakyReLU(negative_slope=0.1),
                layers.Dense(16, kernel_regularizer=regularizers.l2(l2)),
                layers.LeakyReLU(negative_slope=0.1),
                layers.Dense(output_size, activation="linear",
                             kernel_regularizer=regularizers.l2(l2))
            ])
        else:
            try:
                self.nn = load_model(self.load)
            except BaseException:
                print("Fatal! Cannot load model (make sure is .h5 or .keras)")
                sys.exit(1)
        self.nn.compile(
            optimizer=keras.optimizers.Adam(learning_rate=nn_learningrate),
            loss="mse"
        )

    def model_update(self, df):
        """"
        Single db update does not take data aligment
        """
        inp, Y, update = self.update_data_prep(df)
        Ynew = Y + update
        # , sample_weight = weights
        self.nn.fit(inp, Ynew, epochs=1, batch_size=1)

    def action(self, df, epsilon):
        _, Y, _ = self.update_data_prep(df)

        generator = np.random.Generator(np.random.PCG64())
        rnum = generator.random(size=df.height)
        random_action = generator.integers(
            low=0, high=self.output_size, size=df.height)
        actions = np.argmax(Y, axis=1)
        all_actions = pl.DataFrame(
            {"actions": actions, "random_action": random_action,
             "random": rnum})
        all_actions = all_actions.with_columns([
            pl.when(pl.col("random") < epsilon).then(
                pl.col("random_action")).otherwise(
                    pl.col("actions")).alias("final")
        ])
        print("DOWN UP RIGHT LEFT")
        print(f"{Y}")
        return all_actions["final"].to_numpy()

    def action_one_hot(self, df):
        return np.eye(self.output_size)[df["action"]]

    def update_data_prep(self, df):
        """
        All df given must be in the following columns in that order:
        episode_id: unique episode identifier
        action: numeric categorical the actions correspond to the number given
        in the constructor
        rewards: numeric reward for the action
        """
        inp = df[:, 3:].to_numpy()
        Y = self.nn.predict(inp, verbose=0)
        max_next = np.max(Y, axis=1)
        game_ouputs = pl.DataFrame(
            {"episode_id": df["episode_id"], "max_next": max_next})
        game_ouputs = game_ouputs.with_columns(
            pl.col("max_next").shift(-1).over("episode_id").fill_null(0)
        )
        max_next = game_ouputs[:, "max_next"].to_numpy()
        r = df["rewards"].to_numpy()
        w = r + self.gamma * max_next
        comp_reward = np.tile(w[:, np.newaxis], self.output_size)
        onehot = self.action_one_hot(df)
        update = self.alpha * (comp_reward - Y) * onehot
        return (inp, Y, update)

    def replay_train(self, df):
        for i in range(1):
            inp, Y, update = self.update_data_prep(df)
            Ynew = Y + update
            self.nn.fit(inp, Ynew, epochs=1, batch_size=1024)

    def replay_train_individual(self, df):
        episodes = df["episode_id"].unique().sample(fraction=1)
        for episode in tqdm(episodes, desc="Training on replay"):
            df2 = df.filter(pl.col("episode_id") == episode)
            inp, Y, update = self.update_data_prep(df2)
            Ynew = Y + update
            # , sample_weight = weights
            self.nn.fit(inp, Ynew, epochs=1, batch_size=512)

    def save(self):
        self.nn.save(self.savef)

    def loadf(self):
        try:
            self.nn = load_model(self.load)
        except BaseException:
            print("Fatal! Cannot load model (make sure is .h5 or .keras)")
