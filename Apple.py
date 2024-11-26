import pygame

import numpy as np

from game_settings import nsquares, deltax,deltay, RED, GREEN, xmax, xmin, ymax, ymin

class Apple(pygame.sprite.Sprite):
    def __init__(self, posx, posy, color):
        super().__init__()
        self.id = "apple"
        self.color = color
        self.posx = posx
        self. posy = posy
        self.image = pygame.Surface((deltax, deltay))
        if color == "red":
            self.InternalColor = RED
        if color == "green":
            self.InternalColor = GREEN
        self.image.fill(self.InternalColor)
        self.rect = self.image.get_rect(topleft=(xmin + posx*deltax, ymin + posy*deltay))

def random_apple(color):
    x = np.random.randint(0,nsquares)
    y = np.random.randint(0,nsquares)
    newapple = Apple(x,y,color)

    return newapple

def refresh_apples(apples,othercol = None):
    while len(apples) < 3:
        if len ([x for x in apples if x.color == "red"]) < 1:
            newapple = random_apple("red")
        if len ([x for x in apples if x.color == "green"]) < 2:
            newapple = random_apple("green")
        if len(pygame.sprite.spritecollide(newapple, apples,False)) > 0:
            continue
        if othercol is not None and len(pygame.sprite.spritecollide(newapple, othercol,False)) > 0:
            continue
        apples.add(newapple)
