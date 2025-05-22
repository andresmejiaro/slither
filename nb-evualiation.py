# %%
from concurrent.futures import ProcessPoolExecutor
from src.Game import Game
from src.Interpreter import Interpreter
from src.Agent import Agent
from game_settings import alpha, gamma
import numpy as np
import uuid
import polars as pl
# Global model (per process)
global_agent = None

model = "models/newHOTthing.keras"


def init_worker():
    """Executed once per process: loads the model into a global variable."""
    global global_agent
    global_agent = Agent(
        save=None,
        load=model,
        input_size=66,
        output_size=4,
        alpha=alpha,
        gamma=gamma,
        nn_learningrate=0.0
    )


def run_game(_):
    """Run a single game using the global agent and return the played Game object."""
    np.random.seed()
    game = Game()
    Interpreter.agent_loop(agent=global_agent, game=game,
                           epsilon=0, visual=False)
    return {
        "id": str(uuid.uuid4()),  # Unique ID
        "length": len(game.snake),
        "lifetime": len(game.reward_history),
        "last_reward": game.reward_history[-1] if game.reward_history else None
    }


# %%
num_games = 100
num_workers = 4

with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
    results = list(executor.map(run_game, range(num_games)))

# %%

bestresults = pl.DataFrame(results)
# %%

bestresults.write_csv(model+".csv")
# %%


# %%
# models/2ndovernight.keras
