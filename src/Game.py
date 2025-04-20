import pygame
from game_settings import  nsquares, deltax, deltay, WHITE,xmax,xmin, ymax, ymin,  fps, rewards, max_steps
import numpy as np




class Game():
    def __init__(self, headless = False, debug = False, screen = None, machine_mode = False):
       
        self.machine_mode = machine_mode
        self.headless = headless
        self.screen = screen
        if not headless:
            self.clock = pygame.time.Clock()
        self.debug = debug
        self.dead = False
        self.board = np.full((10,10), '0', dtype='<U1')
        self.snakedir = None
        self.n_green_apples = 2
        self.n_red_apples = 1
        self.snake = []
        self.state_history = []
        self.reward_history= []
        self.action_history = []
        ### starting state
        self.start_snake()
        self.add_apples()



    def add_apples(self):
        count_green = np.sum(self.board == 'G')
        count_red = np.sum(self.board == 'R')
        while count_green < self.n_green_apples:
            coord = tuple(np.random.randint(0,9,size=2))
            if self.board[coord] == '0':
               self.board[coord] = 'G'
               count_green+=1 
        while count_red < self.n_red_apples:
            coord = tuple(np.random.randint(0,9,size=2))
            if self.board[coord] == '0':
               self.board[coord] = 'R'
               count_red+=1

    def get_snake_dir_from_n(self,dir_num):
        """Returns the direction of the given num -1 returns the actual direction"""
        match dir_num:
            case 0:
                return np.array([1,0])
            case 1:
                return np.array([-1,0])
            case 2:
                return np.array([0,1])
            case 3:
                return np.array([0,-1])
            case -1:
                return self.snakedir
            case _:
                return None

    def delete_snake(self):
        """Removes the snake from the board state is preserved internally"""
        for pos in self.snake:
            self.board[tuple(pos)] = '0'

    
    def paint_snake(self):
        """Paints the snake on the board"""
        first = True
        for pos in self.snake:
            if first:
                self.board[tuple(pos)] = 'H'
                first =False
            else:   
                self.board[tuple(pos)] = 'S'


    def start_snake(self):
        """Creates the starting position of the snake"""
        done = False
        self.snake = []
        self.snakedir = self.get_snake_dir_from_n(np.random.randint(0,4))
        while not done:
            headpos = np.random.randint(0,9,size=2)
            vect = np.array(self.snakedir)
            last = headpos - 2*vect
            if not (np.max(last) > 9 or np.min(last) <0):
                done = True
        self.snake.append(tuple(headpos))
        self.snake.append(tuple(headpos - vect))
        self.snake.append(tuple(headpos - 2*vect))
        

    def move_snake(self, ndir):
        """"Moves the snake in the given direction -1 keeps the current direction"""
        self.delete_snake()
        if ndir == -1:
            mov_dir = self.snakedir
        else:
            mov_dir = self.get_snake_dir_from_n(ndir)
        if np.array_equal(mov_dir, -self.snakedir) and len(self.snake) > 1:
            mov_dir = self.snakedir 
        next_pos =self.snake[0] + mov_dir
        if max(next_pos) > 9 or min(next_pos)<0:
            self.reward_history.append('W')
            self.dead = True
            return
        next_char = self.board[tuple(next_pos)]
        match next_char:
            case 'G':
                self.snake.insert(0, next_pos)
            case 'R':
                if len(self.snake) > 1:
                    self.snake.insert(0, next_pos)
                    self.snake.pop()
                    self.snake.pop()
                else:
                    self.reward_history.append('R')
                    self.dead = True
                    return
            case '0':
                self.snake.insert(0, next_pos)
                self.snake.pop()
            case 'S':
                self.reward_history.append('S')
                self.dead = True
                return
        self.reward_history.append(next_char)
        self.snakedir = mov_dir
                

        
    def draw_board(self):
        self.screen.fill((0, 0, 0))  # clear screen

        color_map = {
            'H': (255, 255, 255),  # white = head
            'S': (150, 150, 150),  # gray = body
            'G': (0, 255, 0),      # green = green apple
            'R': (255, 0, 0),      # red = red apple
        }

        for i in range(10):
            for j in range(10):
                char = self.board[i, j]
                if char == '0':
                    continue
                color = color_map.get(char, (255, 255, 0))  # fallback yellow for unknown
                pygame.draw.rect(
                    self.screen,
                    color,
                    pygame.Rect(xmin + j * deltax, ymin + i * deltay, deltax, deltay)
                )

        # draw grid overlay
        for it in range(nsquares + 1):
            pygame.draw.line(self.screen, WHITE, (xmin, ymin + it * deltay), (xmax, ymin + it * deltay))
            pygame.draw.line(self.screen, WHITE, (xmin + it * deltax, ymin), (xmin + it * deltax, ymax))

        # handle quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        pygame.display.flip()
        self.clock.tick(fps)


    def cycle(self, action = -1):
        """loop designed to match cycle status -> action->reward"""
        self.state_history.append(self.board)
        self.delete_snake()
        self.action_history.append(self.get_snake_dir_from_n(action))
        self.move_snake(action) #This adds to reward history
        self.paint_snake()
        self.add_apples()




    def loop(self):
        while not self.dead:
            if not self.headless:
                self.draw_board() #se
            if not self.machine_mode:
                action = update_keys()
            else:
                action = self.agent.action
            self.cycle(action)
            if self.debug:
                input("Press enter to continue")
        
    # def loop(self):
    #     self.sna.update_sprites()
    #     refresh_apples(self.apples, self.sna.snake_segments)
    #     if not self.headless:
    #         self.walls.draw(self.screen)
    #         self.draw_background()
    #         self.sna.snake_segments.draw(self.screen)
    #         self.apples.draw(self.screen) 
    #         pygame.display.flip() 
    #     if self.debug:
    #         input("Press enter to continue")
    #     while True:
    #         if self.status is not None:
    #             self.status.collect_board()
    #         if not self.headless:
    #             for event in pygame.event.get():
    #                 if event.type == pygame.QUIT:
    #                     running = False
    #             self.draw_background()
    #         refresh_apples(self.apples,self.sna.snake_segments)
    #         if not self.machine_mode:
    #             keys = update_keys()
    #         else:
    #             keys = self.agent.action(self.status.pl_state)
    #         self.sna.update_state(keys)
    #         self.last_action = self.sna.last_action
    #         self.sna.move()
    #         self.last_reward = rewards["0"]
    #         if self.sna.collided_snake(self.walls) or len(self.sna.segments) == 0 or self.steps == max_steps:
    #             self.last_reward = rewards["W"]
    #             print(f"ded: {self.score} {len(self.sna.segments)}")
    #             self.dead = True
    #         collided_apples = pygame.sprite.groupcollide(self.apples, self.sna.snake_segments, True, False)
    #         for apple in collided_apples.keys():
    #             if apple.color == "red":
    #                 self.sna.setgrowth = -1
    #                 self.last_reward = rewards["R"]
    #             if apple.color == "green":
    #                 self.sna.setgrowth = 1
    #                 self.last_reward = rewards["G"]
    #         self.score += self.last_reward
    #         if self.status is not None:
    #             self.status.update()
    #         print(f"score: {self.score} {len(self.sna.segments)}")
    #         if self.dead:
    #             break
    #         self.steps += 1
    #         if not self.headless:
    #             #move
    #             self.sna.snake_segments.draw(self.screen)
    #             #display
    #             self.apples.draw(self.screen)
    #             pygame.display.flip()
    #             self.clock.tick(fps)
    #         else:
    #             self.clock.tick(0)
    #         if self.debug:
    #             input("Press enter to continue")


    

# Returns a list of all keys and their states
def update_keys():
    keys = pygame.key.get_pressed()
    multi= keys[pygame.K_LEFT] + keys[pygame.K_RIGHT] + keys[pygame.K_DOWN] + keys[pygame.K_UP]
    if multi != 1:
        return -1  
    if keys[pygame.K_LEFT]:
        return 3
    if keys[pygame.K_RIGHT]:
        return 2
    if keys[pygame.K_DOWN]:
        return 0
    if keys[pygame.K_UP]:
        return 1