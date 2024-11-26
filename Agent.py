import numpy as np
import pygame

class Agent():
    def __init__(self,status):
        pass

    def action(self):
        nkey = np.random.randint(0,4)
        keys = {"left":False,
                "right": False,
                "up": False,
                "down": False}
        match nkey:
            case 0:
                keys["left"] = True
            case 1:
                keys["right"] = True
            case 2:
                keys["up"] = True
            case 3:
                keys["down"] = True
            case _:
                pass
        return keys
    
