import pygame
import numpy as np
from game_settings import deltax, deltay, BLUE, xmax,xmin, ymax, ymin

class SnakeSegment(pygame.sprite.Sprite):
    def __init__(self, posx, posy):
        super().__init__()
        self.id = "snake_segment"
        self.posx = posx
        self. posy = posy
        self.image = pygame.Surface((deltax, deltay))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect(topleft=(xmin + posx*deltax, ymin + posy*deltay))


class Snake:
    def __init__(self):
        self.length = 1
        self.segments = []
        self.segments.append(np.array([1,0]))
        self.segments.append(np.array([0,0]))
        self.direction = np.array([1,0])
        self.snake_segments = pygame.sprite.Group()
        self.setgrowth = 0

    def update_sprites(self):
        self.snake_segments.empty()
        for j in self.segments:
            ss = SnakeSegment(*j)
            self.snake_segments.add(ss)


    def move(self):
        self.segments.append( self.segments[-1] + self.direction)
        if self.setgrowth <= 0:
            self.segments.pop(0)
        if self.setgrowth < 0:
            self.segments.pop(0)
        self.setgrowth = 0
        self.update_sprites()
        
    
    def update_state(self,keys):
        new_dir = np.array([0,0])
        if keys[pygame.K_LEFT]:
            new_dir += np.array([-1,0])
        if keys[pygame.K_RIGHT]:
            new_dir += np.array([1,0])
        if keys[pygame.K_DOWN]:
            new_dir += np.array([0,1])
        if keys[pygame.K_UP]:
            new_dir += np.array([0,-1])
        if new_dir.dot(new_dir) != 1:
            return
        if (new_dir + self.direction).dot(new_dir + self.direction) == 0:
            return
        self.direction = new_dir

    def collided_snake(self, walls):
        if len(self.snake_segments) == 0:
            return True
        a = pygame.sprite.groupcollide(self.snake_segments,walls, False, False)
        for seg in self.snake_segments:
            ncol = pygame.sprite.spritecollide(seg, self.snake_segments, False)
            if len(ncol) > 1:
                return True
        if len(a) > 0:
            return True
        return False
