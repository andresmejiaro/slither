from . import Agent
from . import Game
from . import Interpreter, data_augmentation
import pygame
from game_settings import gamma, alpha, nn_learningrate, epsilonstart, epsilonend, nfeatures, screen_height, screen_width, rewards
import random
import datetime


class Bill():
    """
    Bill aka The Snake Charmer is the ruthless leader of the Snake Assassination Squad — the one who calls the shots and charms serpents into submission. A true orchestrator, he commands the training, testing, and exhibition of the AI snake with deadly precision. Whether you're watching a flawless exhibition or throwing your agent into the pit to learn from failure, Bill is behind the scenes, cold and calculated. This isn't just an interface — it's a Kill Bill kind of control.
    """

    def __init__(self, args):
        self.save = args.save
        self.load = args.load
        self.logfile = args.logfile
        self.play = args.play
        self.debug = args.debug
        self.screen = None
        self.sessions = args.sessions
        self.show = args.exhibit
        self.font = None
        self.tlog = args.logtrain

    def run(self):
        if self.play:
            self.run_game()
        elif self.show:
            self.exhibit()
        elif self.tlog:
            self.train_log()

        else:
            self.train()

    def run_game(self):
        self.init_visuals()
        for i in range(self.sessions):
            game = Game(debug=self.debug, screen=self.screen, font=self.font)
            game.loop()
        self.end_visuals()

    def exhibit(self):
        agente = Agent(save=self.save, load=self.load, input_size=nfeatures,
                       output_size=4, alpha=alpha, gamma=gamma, nn_learningrate=nn_learningrate)
        self.init_visuals()
        for i in range(self.sessions):
            game = Game(debug=self.debug, screen=self.screen, font=self.font)
            Interpreter.agent_loop(agente, game, epsilon=0, visual=True)
            if i % 5 == 0 and i != 0:
                agente.loadf()
        self.end_visuals()

    def init_visuals(self):
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.font = pygame.font.SysFont(None, 24)

    def end_visuals(self):
        pygame.quit()

    def train(self):
        agente = Agent(save=self.save, load=self.load, input_size=nfeatures,
                       output_size=4, alpha=alpha, gamma=gamma, nn_learningrate=nn_learningrate)
        for i in range(self.sessions):
            epsilon = epsilonstart + i/self.sessions * \
                (epsilonend - epsilonstart)
            epid = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
            game = Game(debug=self.debug, screen=self.screen)
            Interpreter.agent_loop(agente, game, epsilon=epsilon, visual=False)
            info = game.get_game_info(epid)
            features = Interpreter.load_single_log(info)
            game.export_episode(episode_id=epid, exportfile=self.logfile)
            features = Interpreter.rewards_to_numeric(features, rewards)
            agente.replay_train(features)
            if i % 5 and i != 0:
                agente.save()
        agente.save()

    def train_log(self):
        i = 0
        chunk_size = 1
        agente = Agent(load=self.load, save=self.save, input_size=nfeatures,
                       output_size=4, alpha=alpha, gamma=gamma, nn_learningrate=nn_learningrate)
        for input_data in Interpreter.load_log_file_chunks(self.logfile, chunk_size=chunk_size, skip=0):
            i = i+1
            print(f"training round {i} on {chunk_size} games")
            base = Interpreter.rewards_to_numeric(input_data, rewards)
            agente.replay_train(base)
            if i % 200 == 0:
                agente.save()
