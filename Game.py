import pygame
from Walls import Wall
from Snake import Snake
from Apple import Apple, random_apple, refresh_apples
from game_settings import screen_height, screen_width, nsquares,\
    deltax, deltay, WHITE,xmax,xmin, ymax, ymin, machine_mode


class Game():
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()
        self.walls = pygame.sprite.Group()
        self.apples = pygame.sprite.Group()
        self.create_walls()
        self.sna = Snake(self.walls)

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
        self.walls.draw(self.screen)

        self.draw_background()
        self.sna.update_sprites()
        refresh_apples(self.apples, self.sna.snake_segments)
        self.sna.snake_segments.draw(self.screen)
        self.apples.draw(self.screen) 
        pygame.display.flip() 
        self.status.update()
        while True:
            #Check for external interrupt
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            self.draw_background()
            #Create Apple if eaten orstart of the game
            refresh_apples(self.apples,self.sna.snake_segments)
            #Keyboard controls
            if not machine_mode:
                keys = update_keys()
            else:
                keys = self.agent.action(self.status.pl_state)
            self.sna.update_state(keys)
            self.status.update()
            self.sna.move()
            # death collisions
            if self.sna.collided_snake(self.walls):
                self.status.update()
                print("ded")
                break
            #apple eatng
            collided_apples = pygame.sprite.groupcollide(self.apples, self.sna.snake_segments, True, False)
            for apple in collided_apples.keys():
                if apple.color == "red":
                    self.sna.setgrowth = -1
                if apple.color == "green":
                    self.sna.setgrowth = 1
            #move
            self.sna.snake_segments.draw(self.screen)
            #display
            self.apples.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(5)

    def __del__(self):
        pygame.quit()

    

# Returns a list of all keys and their states
def update_keys():
    a = pygame.key.get_pressed()
    return a  