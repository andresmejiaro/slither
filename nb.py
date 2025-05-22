# %%
import polars as pl

from functools import reduce

# ——— Define your augmentation functions —————————————————————


def horizontal_reflection(df):
    df = df.rename({
        "north_view": "south_view",
        "south_view": "north_view"
    })
    df = df.with_columns(
        pl.col("action").replace(
            {"north": "south", "south": "north"}).alias("action")
    )
    return df


def quarter_turn(df):
    df = df.rename({
        "north_view": "west_view",
        "west_view": "south_view",
        "south_view": "east_view",
        "east_view": "north_view"
    })
    df = df.with_columns(
        pl.col("action").replace(
            {"north": "west", "west": "south", "south": "east", "east": "north"}).alias("action")
    )
    return df


def composition(*funcs):
    return reduce(lambda f, g: lambda x: g(f(x)), funcs)


def add_suffix_to_gameid(df, suffix):
    return df.with_columns(
        (pl.col("game_id") + pl.lit(suffix)).alias("game_id")
    )


def data_augmentation(df):
    # produce all 8 transforms
    df1 = quarter_turn(df)
    df2 = composition(quarter_turn, quarter_turn)(df)
    df3 = composition(quarter_turn, quarter_turn, quarter_turn)(df)
    df4 = horizontal_reflection(df)
    df5 = composition(quarter_turn, horizontal_reflection)(df)
    df6 = composition(quarter_turn, quarter_turn, horizontal_reflection)(df)
    df7 = composition(quarter_turn, quarter_turn,
                      quarter_turn, horizontal_reflection)(df)

    # suffix each
    df1 = add_suffix_to_gameid(df1, "_a")
    df2 = add_suffix_to_gameid(df2, "_b")
    df3 = add_suffix_to_gameid(df3, "_c")
    df4 = add_suffix_to_gameid(df4, "_d")
    df5 = add_suffix_to_gameid(df5, "_e")
    df6 = add_suffix_to_gameid(df6, "_f")
    df7 = add_suffix_to_gameid(df7, "_g")

    # align and concat
    cols = df.columns
    all_dfs = [df, df1, df2, df3, df4, df5, df6, df7]
    aligned = [d.select(cols) for d in all_dfs]
    return pl.concat(aligned)


# ——— Build a tiny mock DataFrame ————————————————————————
df = pl.DataFrame({
    "game_id":   ["g1"],
    "north_view": ["0S00S0S00"],
    "south_view": ["00S00S000"],
    "east_view": ["0S0S0S0S0"],
    "west_view": ["00SS00000"],
    "action":    ["north"]
})

print("\n🟢 Original DF:")
print(df)

# ——— Apply augmentation ——————————————————————————————————
aug = data_augmentation(df)

print("\n🔵 Augmented DF:")
print(aug)


# %%
