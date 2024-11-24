import pygame
import sys
import numpy as np
from Walls import Wall
from Snake import Snake
from Apple import Apple, random_apple, refresh_apples
from game_settings import screen_height, screen_width, nsquares, deltax, deltay, WHITE,xmax,xmin, ymax,ymin
from Status import Status

def draw_background(screen, walls):
    screen.fill((0,0,0))
    
    for it  in range(nsquares + 1    ):
        pygame.draw.line(screen, WHITE, (xmin, ymin + it*deltay), (xmax, ymin + it *deltay) )
        pygame.draw.line(screen, WHITE, (xmin + it*deltax,ymin), (xmin + it *deltax,ymax) )

    walls.draw(screen)

def update_keys():
    return pygame.key.get_pressed()  # Returns a list of all keys and their states

def main():
    #init parameters
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    walls = pygame.sprite.Group()
    apples = pygame.sprite.Group()
    sna = Snake()

    # Create walls
    for i in range( nsquares + 2):
        s1 = Wall(-1, (i-1)  )
        s2 = Wall(nsquares, (i-1))
        s3 = Wall((i-1), -1)
        s4 = Wall((i-1), nsquares)
        for w in [s1,s2,s3,s4]:
            walls.add(w)

    #Set status monitos
    status = Status(sna,apples,walls)
    #Main loop
    running = True
    while running:
        #Check for external interrupt
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        draw_background(screen,walls)
        #Create Apple if eaten orstart of the game
        refresh_apples(apples,sna.snake_segments)
        #Keyboard controls
        keys = update_keys()
        sna.update_state(keys)
        sna.move()
        # death collisions
        if sna.collided_snake(walls):
            print("ded")
            break
        #apple eatng
        collided_apples = pygame.sprite.groupcollide(apples,sna.snake_segments,True, False)
        for apple in collided_apples.keys():
            if apple.color == "red":
                sna.setgrowth = 1
            if apple.color == "green":
                sna.setgrowth = -1
        #move
        status.update()

        sna.snake_segments.draw(screen)
        #display
        apples.draw(screen)
        pygame.display.flip()
        clock.tick(5)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()