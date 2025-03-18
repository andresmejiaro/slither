import sys
import numpy as np
from Status import Status
from Agent import Agent
from Game import Game
from game_settings import machine_mode


def main():
    
    game = Game()
    #Set status monitor
    status = Status(game)
    game.set_status(status)
    ## Start Agent
    if  machine_mode:
        agent = Agent()
        agent.set_status(status)
        game.set_agent(agent)
    #Main loop
    game.loop()
    agent.model_update()
    sys.exit()

if __name__ == "__main__":
    main()