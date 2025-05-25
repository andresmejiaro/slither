import numpy as np
import polars as pl
import json
from Agent import Agent
from Game import Game
from tqdm import tqdm


class Interpreter():
    def __init__(self):
        pass

    @staticmethod
    def extract_views_from_board(mat):
        """
        Takes a board and extracts the views needed as per the subject.
        mat should be a matrix with characters from the snake game as given
        """
        assert np.sum(
            mat == 'H') == 1, "Board should have one snake with one head"
        headpos = np.argwhere(mat == 'H')[0]

        horviews = mat[headpos[0], :]

        horviews_text = "".join(horviews)
        west_view, east_view = horviews_text.split("H")
        west_view = west_view[::-1]
        vertviews = mat[:, headpos[1]]
        vertviews_text = "".join(vertviews)
        north_view, south_view = vertviews_text.split("H")
        north_view = north_view[::-1]
        # padding with "W" because the wall object is the limit of the matrix
        return pl.DataFrame({"north_view": north_view + 'W',
                             "south_view": south_view + 'W',
                             "east_view": east_view + 'W',
                             "west_view": west_view + 'W'})

    @staticmethod
    def build_inputs_from_views(df):
        """
        This takes a matrix of views (see extract views from board)
        and creates features for the Agent. Note that building
        from views makes sure we comply with the required no extra info
        used or needed
        """

        # distance to closest one
        df = df.with_columns([
            pl.col(direction).str.find(
                char).fill_null(-1).alias(f"{direction}_{char}_input")
            for direction in ["north_view", "south_view", "east_view",
                              "west_view"]
            for char in ["W", "S", "R", "G"]
        ])
        # inverse distance to closest one
        df = df.with_columns([
            pl.when(pl.col(f"{direction}_{char}_input") >= 0).then(
                1 / (1 + pl.col(f"{direction}_{char}_input"))).otherwise(
                    -1).alias(f"{direction}_{char}_inv_input")
            for direction in ["north_view", "south_view", "east_view",
                              "west_view"]
            for char in ["W", "S", "R", "G"]
        ])
        # boolean is something there
        df = df.with_columns([
            pl.col(f"{direction}_{char}_input").lt(
                0).alias(f"{direction}_{char}_exists_input")
            for direction in ["north_view", "south_view", "east_view",
                              "west_view"]
            for char in ["S", "R", "G"]
        ])
        # blocked dirrection you shall not pass
        df = df.with_columns([
            ((pl.col(f"{direction}_S_input") == 0) |
                (pl.col(f"{direction}_W_input") == 0)).cast(
                pl.Int8).alias(f"{direction}_blocked_input")
            for direction in ["north_view", "south_view", "east_view",
                              "west_view"]
        ])
        # count of object in direction
        df = df.with_columns([
            pl.col(direction).str.count_matches(char).alias(
                f"{direction}_{char}_count_input")
            for direction in ["north_view", "south_view", "east_view",
                              "west_view"]
            for char in ["S", "G"]
        ])
        # count of object global
        df = df.with_columns([
            pl.sum_horizontal([
                pl.col(f"{direction}_{char}_count_input")
                for direction in ["north_view", "south_view", "east_view",
                                  "west_view"]
            ]).alias(f"{char}_count_input")
            for char in ["S", "G"]
        ])
        # min distance across all direction / max inverse distance across all
        # dir
        df = df.with_columns([
            pl.min_horizontal([
                pl.col(f"{direction}_{char}_input")
                for direction in ["north_view", "south_view", "east_view",
                                  "west_view"]
            ]).alias(f"{char}_min_dist_input")
            for char in ["W", "S", "R", "G"]
        ])
        df = df.with_columns([
            pl.max_horizontal([
                pl.col(f"{direction}_{char}_inv_input")
                for direction in ["north_view", "south_view", "east_view",
                                  "west_view"]
            ]).alias(f"{char}_max_inv_input")
            for char in ["W", "S", "R", "G"]
        ])

        #########
        dirs = ["north_view", "south_view", "east_view", "west_view"]

        # 2nd 'S'
        for d in dirs:
            pos1 = pl.col(d).str.find("S").fill_null(-1)
            df = df.with_columns(
                pl.when(pos1 >= 0)
                .then(
                    pl.col(d)
                    .str.tail(-(pos1 + 1))
                        .str.find("S")
                        .fill_null(-1)
                )
                .otherwise(-1)
                .alias(f"{d}_S_2nd_rel")
            )

        # 3rd 'S'
        for d in dirs:
            pos1 = pl.col(d).str.find("S").fill_null(-1)
            pos2 = pl.col(f"{d}_S_2nd_rel")
            df = df.with_columns(
                pl.when((pos1 >= 0) & (pos2 >= 0))
                .then(
                    pl.col(d)
                    .str.tail(-(pos1 + 1 + pos2 + 1))
                        .str.find("S")
                        .fill_null(-1)
                )
                .otherwise(-1)
                .alias(f"{d}_S_3rd_rel")
            )

        # 4th 'S'
        for d in dirs:
            pos1 = pl.col(d).str.find("S").fill_null(-1)
            pos2 = pl.col(f"{d}_S_2nd_rel")
            pos3 = pl.col(f"{d}_S_3rd_rel")
            df = df.with_columns(
                pl.when((pos1 >= 0) & (pos2 >= 0) & (pos3 >= 0))
                .then(
                    pl.col(d)
                    .str.tail(-(pos1 + 1 + pos2 + 1 + pos3 + 1))
                        .str.find("S")
                        .fill_null(-1)
                )
                .otherwise(-1)
                .alias(f"{d}_S_4th_rel")
            )

        ############

        return df[:, 4:]

    @staticmethod
    @staticmethod
    def print_view(view: pl.DataFrame):
        assert view.height == 1, "Pass a single view"
        assert all(
            col in view.columns for col in [
                "north_view", "south_view", "east_view", "west_view"]), \
            "Make sure to pass a view to this"
        northview = view["north_view"][0]
        southview = view["south_view"][0]
        eastview = view["east_view"][0]
        westview = view["west_view"][0]
        z = ''
        px = len(westview)
        py = len(northview)
        nsquares = px + len(eastview) + 1
        for a in range(nsquares + 2):
            for b in range(nsquares + 2):
                if a != py and b != px:
                    z += ' '
                    continue
                if a < py:
                    try:
                        z += northview[-a - 1]
                    except BaseException:
                        z += 'W'
                    continue
                if a > py:
                    try:
                        z += southview[a - py - 1]
                    except BaseException:
                        z += 'W'
                    continue
                if b < px:
                    try:
                        z += westview[-b - 1]
                    except BaseException:
                        z += 'W'
                    continue
                if b > px:
                    try:
                        z += eastview[b - px - 1]
                    except BaseException:
                        z += 'W'
                    continue
                z += 'H'

            z += '\n'
        print(z)

    @staticmethod
    def agent_loop(agent: Agent, game: Game, epsilon=0.1, visual=False):
        """Runs a game with the given agent giving the commands.
        If visual is true make sure that the game has a screen attached
        """
        while not game.dead:
            if visual:
                game.draw_board()
            boiler = pl.DataFrame(
                {"episode_id": "generic", "action": -1, "rewards": -1})
            views = Interpreter.extract_views_from_board(game.board)
            features = Interpreter.build_inputs_from_views(views)
            Interpreter.print_view(views)
            action = agent.action(boiler.hstack(features), epsilon)
            Interpreter.print_dir(action)
            print(f"Epsilon {epsilon}")
            game.cycle(action)

    @staticmethod
    def print_dir(action):
        match action:
            case 0:
                print("DOWN")
            case 1:
                print("UP")
            case 2:
                print("RIGHT")
            case 3:
                print("LEFT")

    @staticmethod
    def load_single_log(jdict) -> pl.DataFrame:
        """ Turns a single log into info passable to the agent"""
        inputs = [
            Interpreter.build_inputs_from_views(
                Interpreter.extract_views_from_board(
                    np.array(x))) for x in jdict["states"]]
        inputs = pl.concat(inputs)
        core = pl.DataFrame({"episode_id": jdict["episode_id"],
                             "action": jdict["actions"],
                             "rewards": jdict["rewards"]})
        return core.hstack(inputs)

    @staticmethod
    def load_log_file(file) -> pl.DataFrame:
        """ loads a logfile to train"""
        with open(file) as f:
            data = [Interpreter.load_single_log(json.loads(
                line)) for line, _ in zip(tqdm(f), range(20))]
        return pl.concat(data)

    @staticmethod
    def load_log_file_chunks(file, chunk_size=10, skip=0) -> pl.DataFrame:
        """ loads a logfile to train"""
        linecount = 0
        with open(file) as f:
            buffer = []
            for line in f:
                linecount += 1
                skip = 290
                if linecount < skip:
                    continue
                data = Interpreter.load_single_log(json.loads(line))
                data2 = data.filter(pl.col("rewards") == 'G')
                w = data2.height
                if w < 35:
                    print(f"skipping record {linecount} reward: {w}")
                    continue
                buffer.append(data)
                if len(buffer) >= chunk_size:
                    yield pl.concat(buffer)
                    buffer = []
            if buffer:
                yield pl.concat(buffer)

    @staticmethod
    def rewards_to_numeric(
            df: pl.DataFrame,
            rewards_dict: dict) -> pl.DataFrame:
        return df.with_columns([
            pl.col("rewards").replace(rewards_dict).cast(pl.Int64)
        ])
