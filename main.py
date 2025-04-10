from Status import Status
from Agent import Agent
from Game import Game
import argparse
import sys
from game_settings import temperature, epsilonend, epsilonstart
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="I'm a snake")
    parser.add_argument("-save", type=str, default="models/latest.keras", help="Save model route")
    parser.add_argument("-load", type=str, default="models/latest.keras", help="Load model route")
    parser.add_argument("-dontlearn",action="store_true",help="Don't update model on each training")
    parser.add_argument("-headless",action="store_true", help="Dont' show game while training")
    parser.add_argument("-sessions", type=int, help="number of training sessions", default=10)
    parser.add_argument("-play",action="store_true", help="Play the game")   
    return parser.parse_args()


def training(args):
    agent = Agent(save = args.save, load = args.load)
    if args.dontlearn:
        agent.epsilon = 0
    for i in range(args.sessions):
    #    if i % 1 == 0:
        if args.dontlearn and i % 10 == 0 and i != 0:
            agent = Agent(save = args.save, load = args.load)
            agent.epsilon = 0
        agent.temperature = 1
    #    else:
    #        agent.temperature = temperature - (temperature - 1)*i/args.sessions
        if not args.dontlearn:
            agent.epsilon = epsilonstart - (epsilonstart - epsilonend)*i/args.sessions
        game = Game(headless = args.headless)
        game.sna.machine_mode = True
        game.machine_mode = True
        status = Status(game)
        game.set_status(status)
        agent.set_status(status)
        game.set_agent(agent)
        game.loop()
        print(f"in game {i + 1} of {args.sessions}")
        if not args.dontlearn:
            agent.model_update()
            if i % 5 == 4 or i == args.sessions - 1:
                print("saving model....")
                print(f"Saving to {args.save}")
                try:
                    agent.nn.save(args.save)
                except:
                    print("Fatal! Cannot save model (make sure is .h5 or .keras)")
                    sys.exit(1)
            if i % 5 == 0 and i != 0:
                print("retraining on best 10 models")
                agent.replay_train()
        del game
        del status

def main():
    
    args = parse_args()

    if args.play:
        game = Game()
        game.sna.machine_mode = False
        game.machine_mode = False
        game.loop()  
    else:
        training(args)

    
    
if __name__ == "__main__":
    main()