import polars as pl
import numpy as np
from functools import reduce

#%%
def add_action(df):
    df2 = df.with_columns(
    pl.when(
        (df["north_view"].str.len_chars().cast(pl.Int32) - df["north_view"].str.len_chars().shift(1).cast(pl.Int32)) == 1
    ).then(pl.lit("south"))
    .when(
        (df["north_view"].str.len_chars().cast(pl.Int32) - df["north_view"].str.len_chars().shift(1).cast(pl.Int32)) == -1
    ).then(pl.lit("north"))
    .when(
        (df["west_view"].str.len_chars().cast(pl.Int32) - df["west_view"].str.len_chars().shift(1).cast(pl.Int32)) == 1
    ).then(pl.lit("east"))
    .when(
        (df["west_view"].str.len_chars().cast(pl.Int32) - df["west_view"].str.len_chars().shift(1).cast(pl.Int32)) == -1
    ).then(pl.lit("west"))
    .otherwise(None)  # No movement detected
    .alias("direction")
    )

    df2 = df2.with_columns(
    pl.when(
        df["north_view"].str.len_chars() + df["south_view"].str.len_chars() == df["east_view"].str.len_chars() + df["west_view"].str.len_chars()
    ).then(pl.col("direction"))
    .when(df["north_view"].str.len_chars() > 1).then(pl.lit("south"))
    .when(df["south_view"].str.len_chars() > 1).then(pl.lit("north"))
    .when(df["east_view"].str.len_chars() > 1).then(pl.lit("west"))
    .when(df["west_view"].str.len_chars() > 1).then(pl.lit("east"))
    .otherwise(pl.col("direction"))
    .alias("direction")
    )
    return df2

def add_event(df):
    return df.with_columns([
        pl.when(pl.col("direction") == "north").then(pl.col("north_view").shift(1).fill_null("N").str.slice(0, 1))
        .when(pl.col("direction") == "south").then(pl.col("south_view").shift(1).fill_null("N").str.slice(0, 1))
        .when(pl.col("direction") == "east").then(pl.col("east_view").shift(1).fill_null("N").str.slice(0, 1))
        .when(pl.col("direction") == "west").then(pl.col("west_view").shift(1).fill_null("N").str.slice(0, 1))
        .otherwise(None)
        .alias("event")
    ])

def first_char(s: str, char: str)-> int:
    return s.find(char) 
   
def add_closest(df):
    return df.with_columns([
        pl.col(direction).str.find(char).fill_null(-1).alias(f"{direction}_{char}")
        for direction in ["north_view", "south_view", "east_view", "west_view"]
        for char in ["W", "S", "R", "G"]
    ])

def add_closest_sl(df):
    #df = df.vstack(df)
    df = add_closest(df)
    #df = df[1,:]
    return df 

def preprocess_games(df):
    df = df.group_by("game_id").map_groups(add_closest)
    return df


def horizontal_reflection(df):
    df = df.rename({
        "north_view":"south_view",
        "south_view": "north_view"
    })
    changes = {
        "north":"south",
        "south": "north"
    }
    df = df.with_columns(
        pl.col("action").replace(changes).alias("col")
    )

    return df
    

def quarter_turn(df):
    df = df.rename({
        "north_view":"west_view",
        "west_view": "south_view",
        "south_view":"east_view",
        "east_view": "north_view"
    })
    changes = {
        "north":"west",
        "west": "south",
        "south":"east",
        "east": "north"
    }
    df = df.with_columns(
        pl.col("action").replace(changes).alias("col")
    )

    return df

def composition(*funcs):
    return reduce(lambda f, g: lambda x: g(f(x)),funcs)

def add_sufix_to_gameid(df, suffix):
    return df.with_columns(
        pl.col("game_id")+pl.lit(suffix)
    )


def data_augmentation(df):
    return df
    df1 = quarter_turn(df)
    df2 = composition(quarter_turn, quarter_turn)(df)
    df3 = composition(quarter_turn, quarter_turn,quarter_turn)(df)
    df4 = horizontal_reflection(df)
    df5 = composition(quarter_turn, horizontal_reflection)(df)
    df6 = composition(quarter_turn, quarter_turn, horizontal_reflection)(df)
    df7 = composition(quarter_turn,quarter_turn,quarter_turn, horizontal_reflection)(df)
    df1 = add_sufix_to_gameid(df1,"a")
    df2 = add_sufix_to_gameid(df2,"b")
    df3 = add_sufix_to_gameid(df3,"c")
    df4 = add_sufix_to_gameid(df4,"d")
    df5 = add_sufix_to_gameid(df5,"e")
    df6 = add_sufix_to_gameid(df6,"f")
    df7 = add_sufix_to_gameid(df7,"g")
    cols_order = df.columns  
    dfs_aligned = [d.select(cols_order) for d in [df, df1, df2, df3, df4, df5, df6, df7]]
    return pl.concat(dfs_aligned)


def load_filter_data():
    logs = pl.read_csv("snakelogs.csv")
    return logs

def add_reward(df,rewards):
    return df.with_columns([
            pl.col("event").replace(rewards).cast(pl.Int32).alias("rewards")
        ])

def align_status_rewards(df):
    df = df.with_columns(pl.col("event").shift(-1), pl.col("direction").shift(-1))
    df = df.filter(pl.col("event").is_not_null())
    return df

def direction_one_hot(df):
    action_categories = np.array(["north", "south", "east", "west"])
    actions = df["action"].to_numpy()

    matches = actions[:, None] == action_categories  
    indices = matches.argmax(axis=1)

    onehot = np.eye(len(action_categories))[indices]
    return onehot