import pygame

from game_settings import deltax, deltay, WHITE, xmin, xmax, ymin, ymax


class Wall(pygame.sprite.Sprite):
    
    def __init__(self, posx, posy):
        super().__init__()
        self.id = "wall"
        self.posx = posx
        self. posy = posy
        self.image = pygame.Surface((deltax, deltay))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect(topleft=(xmin + posx*deltax, ymin + posy*deltay))
