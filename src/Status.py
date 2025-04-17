import pygame, csv, os, datetime 
from game_settings import *
import polars as pl
import numpy as np
from . import SnakeSegment
from . import add_closest_sl

class Status():
    #def __init__(self, snake, apples, walls):
    def __init__(self, game):
        
        # logging related stuff
        self.log_filename="snakelogs.csv"
        self.file_exists = os.path.isfile(self.log_filename)
        self.log_file = open(self.log_filename,"a", newline="")
        self.writer = csv.writer(self.log_file)
        if not self.file_exists:
            self.writer.writerow(["game_id", "turn", "north_view", "south_view", "east_view", "west_view", "action", "reward"])
            self.log_file.flush()
            
        self.new_game(game)
        self.game = game
        


    def new_game(self,game):
        #game related stuff
        self.snake_len = 0
        self.northview = []
        self.southview = []
        self.westview = []
        self.eastview = []
        self.pos = np.array([0,0])
        self.snake = game.sna
        self.apples = game.apples
        self.walls = game.walls
        
        schema = {
            "game_id": pl.Utf8,  # String type
            "turn": pl.Int64,     # Integer type
            "north_view": pl.Utf8,
            "south_view": pl.Utf8,
            "east_view": pl.Utf8,
            "west_view": pl.Utf8,
            "action": pl.String,
            "reward": pl.Int64
        }
        self.pl_state = pl.DataFrame(schema=schema)
        self.last_game = pl.DataFrame(schema=schema)
        self.game_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.turn = -1
        self.first = True

    
    def collect_board(self):
        if len(self.snake.segments) > 0:
            self.pos = self.snake.segments[-1]
        self.northview = self.view(np.array([0,-1]))
        self.southview = self.view(np.array([0,1]))
        self.eastview = self.view(np.array([1,0]))
        self.westview = self.view(np.array([-1,0]))
        self.pl_state = pl.DataFrame([{
            "game_id": self.game_id,
            "turn": self.turn,
            "north_view":"".join( self.northview),
            "south_view":"".join( self.southview),
            "east_view": "".join(self.eastview),
            "west_view": "".join(self.westview),
            "action": "",
            "reward": 0
        }])
    
    
    
    
    def update(self):
        self.turn += 1        
        self.pl_state["action"][0]= self.game.last_action,
        self.pl_state["reward"][0]= self.game.last_reward
        self.printview()
        self.last_game = self.last_game.vstack(self.pl_state)
        self.logState()
        

    def logState(self):
        self.writer.writerow([self.game_id, self.turn,''.join(self.northview),''.join(self.southview),''.join(self.eastview),''.join(self.westview), self.game.last_action, self.game.last_reward])

    def view(self, direction = np.array([1,0])):
        response = []
        offset = 1

        if len(pygame.sprite.spritecollide(SnakeSegment(*self.pos), self.walls, False)) > 0:
            northcond = (self.pos[1] == -1) and (np.array_equal(direction,np.array([0,-1])))
            southcond = (self.pos[1] == nsquares) and (np.array_equal(direction,np.array([0,1])))
            westcond = (self.pos[0] == -1) and (np.array_equal(direction,np.array([-1,0])))
            eastcond = (self.pos[0] == nsquares) and (np.array_equal(direction,np.array([1,0])))
            if northcond or southcond or westcond or eastcond:
                return ["W"]

        while True:
            tempSprite = SnakeSegment(*(self.pos + offset * direction))
            offset += 1
            if len(pygame.sprite.spritecollide(tempSprite,self.snake.snake_segments,False)):
                response.append('S')
                continue
            applescol = pygame.sprite.spritecollide(tempSprite,self.apples, False)
            if len(applescol) > 0:
                    response.append(applescol[0].color[0].upper())
                    continue
            wallcol = pygame.sprite.spritecollide(tempSprite, self.walls,False)
            if len(wallcol) > 0:
                response.append('W')
                break
            response.append('0')
            
        return response

    def printview(self):
        colided = False
        z = ''    
        px = len(self.westview)
        py = len(self.northview)
        px2 = len(self.eastview)
        py2 = len(self.southview)
        if px + px2 != py + py2:
            print("colided")
            colided = True
        for a in range(nsquares + 2):
            for b in range(nsquares + 2):
                if a != py and b != px:
                    z += ' '
                    continue
                if a < py:
                    try:
                        z += self.northview[-a -1]
                    except:
                        z += 'W'
                    continue
                if a > py:
                    try:
                        z += self.southview[a - py - 1]
                    except:
                        z += 'W'
                    continue
                if b < px:
                    try:
                        z += self.westview[-b -1]
                    except:
                        z += 'W'
                    continue
                if b > px:
                    try:
                        z += self.eastview[b - px - 1]
                    except:
                        z += 'W'
                    continue
                z += 'H'
                
            z += '\n'
        print(z)

    def close(self):
        self.log_file.flush()
        self.log_file.close()
    