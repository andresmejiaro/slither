import pygame
from . import Wall
from . import Snake
from . import Apple, random_apple, refresh_apples
from game_settings import  nsquares, deltax, deltay, WHITE,xmax,xmin, ymax, ymin,  fps, rewards, max_steps


class Game():
    def __init__(self, headless = False, debug = False, screen = None):
       
        self.headless = headless
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.walls = pygame.sprite.Group()
        self.apples = pygame.sprite.Group()
        self.create_walls()
        self.sna = Snake(self.walls)
        self.status = None
        self.score = 0
        self.steps = 0
        self.debug = debug
        self.last_action = ""
        self.last_reward = 0
        self.dead = False
        

    def set_status(self,status):
        self.status = status
    
    def set_agent(self, agent):
        self.agent = agent
    
    def create_walls(self):
        for i in range( nsquares + 2):
            s1 = Wall(-1, (i-1)  )
            s2 = Wall(nsquares, (i-1))
            s3 = Wall((i-1), -1)
            s4 = Wall((i-1), nsquares)
            for w in [s1,s2,s3,s4]:
                self.walls.add(w)
        

    def draw_background(self):
        self.screen.fill((0,0,0))
        for it  in range(nsquares + 1    ):
            pygame.draw.line(self.screen, WHITE, (xmin, ymin + it*deltay), (xmax, ymin + it *deltay) )
            pygame.draw.line(self.screen, WHITE, (xmin + it*deltax,ymin), (xmin + it *deltax,ymax) )


    def loop(self):
        self.sna.update_sprites()
        refresh_apples(self.apples, self.sna.snake_segments)
        if not self.headless:
            self.walls.draw(self.screen)
            self.draw_background()
            self.sna.snake_segments.draw(self.screen)
            self.apples.draw(self.screen) 
            pygame.display.flip() 
        
        while True:
            if self.status is not None:
                self.status.collect_board()
            if not self.headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                self.draw_background()
            refresh_apples(self.apples,self.sna.snake_segments)
            if not self.machine_mode:
                keys = update_keys()
            else:
                keys = self.agent.action(self.status.pl_state)
            self.sna.update_state(keys)
            self.last_action = self.sna.last_action
            self.sna.move()
            self.last_reward = rewards["0"]
            if self.sna.collided_snake(self.walls) or len(self.sna.segments) == 0 or self.steps == max_steps:
                self.last_reward = rewards["W"]
                print(f"ded: {self.score} {len(self.sna.segments)}")
                self.dead = True
            collided_apples = pygame.sprite.groupcollide(self.apples, self.sna.snake_segments, True, False)
            for apple in collided_apples.keys():
                if apple.color == "red":
                    self.sna.setgrowth = -1
                    self.last_reward = rewards["R"]
                if apple.color == "green":
                    self.sna.setgrowth = 1
                    self.last_reward = rewards["G"]
            self.score += self.last_reward
            if self.status is not None:
                self.status.update()
            print(f"score: {self.score} {len(self.sna.segments)}")
            if self.dead:
                break
            self.steps += 1
            if not self.headless:
                #move
                self.sna.snake_segments.draw(self.screen)
                #display
                self.apples.draw(self.screen)
                pygame.display.flip()
                self.clock.tick(fps)
            else:
                self.clock.tick(0)
            if self.debug:
                input("Press enter to continue")


    

# Returns a list of all keys and their states
def update_keys():
    a = pygame.key.get_pressed()
    return a  