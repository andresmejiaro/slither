from src.Bill import Bill
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="I'm a snake")
    parser.add_argument("-play",action="store_true", help="Play the game ignores other options")
    parser.add_argument("-logtrain", action="store_true", help="train from logfile may ignore other options")
    parser.add_argument("-exhibit",action="store_true",help="Shows model in autorun may ignore other options")
    parser.add_argument("-save", type=str, default="models/latest.keras", help="Save model route")
    parser.add_argument("-load", type=str, default="models/latest.keras", help="Load model route")
    parser.add_argument("-logfile", type=str, default="logs/snakelog.jsonl", help="Logfile route")
    parser.add_argument("-sessions", type=int, help="number of training sessions", default=10)
    parser.add_argument("-debug", action="store_true", help="Stop at each play")
    

    return parser.parse_args()


# def training(args):
#     agent = Agent(save = args.save, load = args.load)
#     if args.dontlearn:
#         agent.epsilon = 0
#     if not args.headless:
#         pygame.init()
#         screen = pygame.display.set_mode((screen_width, screen_height))
#     else:
#         screen = pygame.display.set_mode((1,1), pygame.HIDDEN)
#     for i in range(args.sessions):
#     #    if i % 1 == 0:
#         if args.dontlearn and i % 10 == 0 and i != 0:
#             agent = Agent(save = args.save, load = args.load)
#             agent.epsilon = 0
#         if not args.dontlearn:
#             agent.epsilon = epsilonstart - (epsilonstart - epsilonend)*i/args.sessions
#         game = Game(headless = args.headless, debug=args.debug, screen = screen)
#         game.sna.machine_mode = True
#         game.machine_mode = True
#         status = Status(game, args.logfile)
#         game.set_status(status)
#         agent.set_status(status)
#         game.set_agent(agent)
#         game.loop()
#         print(f"in game {i + 1} of {args.sessions}")
#         if not args.dontlearn:
#             agent.model_update()
#             if i % 5 == 4 or i == args.sessions - 1:
#                 print("saving model....")
#                 print(f"Saving to {args.save}")
#                 try:
#                     agent.nn.save(args.save)
#                 except:
#                     print("Fatal! Cannot save model (make sure is .h5 or .keras)")
#                     sys.exit(1)
#         else:
#             if i % 5 == 0 and i != 0:
#                 try:
#                     agent.nn = load_model(args.load)
#                     print("updated model")
#                 except:
#                     print("skipping model load") 
#         status.close()
#         del game
#         del status
#     if not args.headless:
#         pygame.quit()

def main():
    
    args = parse_args()
    
    Bill(args).run()

    # if args.play:
    #     pygame.init()
    #     screen =         screen = pygame.display.set_mode((screen_width, screen_height))
    #     while True:
    #         game = Game(screen = screen, debug= args.debug)
    #         game.loop() 
    #     pygame.quit() 
    # else:
    #     training(args)

    
    
if __name__ == "__main__":
    main()