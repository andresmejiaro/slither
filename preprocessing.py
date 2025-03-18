#%%
import polars as pl


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
        pl.col(direction).shift(1).str.find(char).fill_null(-1).alias(f"{direction}_{char}")
        for direction in ["north_view", "south_view", "east_view", "west_view"]
        for char in ["W", "S", "R", "G"]
    ])

def add_closest_sl(df):
    df = df.vstack(df)
    df = add_closest(df)
    df = df[1,:]
    return df 

def preprocess_games(df):
    df = df.group_by("game_id").map_groups(add_action)
    df = df.group_by("game_id").map_groups(add_event)
    df = df.group_by("game_id").map_groups(add_closest)
    return df


