import numpy as np
from Snake import SnakeSegment
import pygame 
from game_settings import nsquares

class Status():
    def __init__(self, snake, apples, walls):
        self.snake_len = 0
        self.northview = []
        self.southview = []
        self.westview = []
        self.eastview = []
        self.score = 0
        self.pos = np.array([0,0])
        self.snake = snake
        self.apples = apples
        self.walls = walls
        

    def update(self):
        self.snake_len = len(self.snake.snake_segments)
        self.pos = self.snake.segments[-1]
        self.northview = self.view(np.array([0,-1]))
        self.southview = self.view(np.array([0,1]))
        self.eastview = self.view(np.array([1,0]))
        self.westview = self.view(np.array([-1,0]))        
        self.printview()
        

    def view(self, direction = np.array([1,0])):
        response = []
        wallyet = True
        offset = 1
        while wallyet:
            tempSprite = SnakeSegment(*(self.pos + offset * direction))
            offset += 1
            if len(pygame.sprite.spritecollide(tempSprite,self.snake.snake_segments,False)):
                response.append('s')
                continue
            applescol = pygame.sprite.spritecollide(tempSprite,self.apples, False)
            if len(applescol) > 0:
                    response.append(applescol[0].color[0])
                    continue
            wallcol = pygame.sprite.spritecollide(tempSprite, self.walls,False)
            if len(wallcol) > 0:
                response.append('w')
                break
            response.append(' ')
            
        return response

    def printview(self):
        z = ''
        px = len(self.westview)
        py = len(self.northview)
        for a in range(nsquares + 2):
            for b in range(nsquares + 2):
                if a != py and b != px:
                    z += 'o'
                    continue
                if a < py:
                    z += self.northview[-a -1]
                    continue
                if a > py:
                    z += self.southview[a - py - 1]
                    continue
                if b < px:
                    z += self.westview[-b -1]
                    continue
                if b > px:
                    z += self.eastview[b - px - 1]
                    continue
                z += 'x'
                
            z += '\n'
        print(z)
    