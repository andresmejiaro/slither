import pygame
import numpy as np
from game_settings import deltax, deltay, BLUE, xmax,xmin, ymax, ymin, nsquares

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
        self.segments = []
        #check = False
        nrot = np.random.randint(0,3)
        rot_mat= np.array([[0,-1],[1,0] ])
        self.direction = np.linalg.matrix_power(rot_mat,nrot).dot(np.array([1,0]))
        if self.direction[0] > 0:
            head_x = np.random.randint(3,nsquares)
        elif self.direction[0] < 0:
            head_x = np.random.randint(0,nsquares-3)
        else:
            head_x = np.random.randint(0,nsquares)
        if self.direction[1] > 0:
            head_y = np.random.randint(3,nsquares)
        elif self.direction[1] < 0:
            head_y = np.random.randint(0,nsquares-3)
        else:
            head_y = np.random.randint(0,nsquares)
        
        #print(f"headx {head_x} heady{head_y} direction {self.direction}")
        #    check = max(np.array([head_x,head_y])-2*self.direction) < nsquares and min(np.array([head_x,head_y])-2*self.direction) >=0
        
        self.segments.append(np.array([head_x,head_y]))
        self.segments.append(np.array([head_x,head_y])-self.direction)
        self.segments.append(np.array([head_x,head_y])-self.direction-self.direction)
        self.snake_segments = pygame.sprite.Group()
        self.setgrowth = 0
        self.last_action = ""

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
        if self.machine_mode:
            if keys["left"]:
                new_dir += np.array([-1,0])
                self.last_action = "west"
            if keys["right"]:
                new_dir += np.array([1,0])
                self.last_action = "east"
            if keys["down"]:
                new_dir += np.array([0,1])
                self.last_action = "south"
            if keys["up"]:
                new_dir += np.array([0,-1])
                self.last_action = "north"
        else:
            if keys[pygame.K_LEFT]:
                new_dir += np.array([-1,0])
                self.last_action = "west"
            if keys[pygame.K_RIGHT]:
                new_dir += np.array([1,0])
                self.last_action = "east"
            if keys[pygame.K_DOWN]:
                new_dir += np.array([0,1])
                self.last_action = "south"
            if keys[pygame.K_UP]:
                new_dir += np.array([0,-1])
                self.last_action = "north"
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
            if len(ncol) > 2:
                return True
        if len(a) > 0:
            return True
        return False
