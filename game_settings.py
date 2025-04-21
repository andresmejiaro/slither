## user modifiable

screen_width, screen_height = 800, 600
nsquares = 10
fps = 5
max_steps = 500
rewards = {"0": -1, "W": -1000, "R": -100, "G": 500, "S": -1001}

# Training hyperparams
gamma = 0.9
alpha = 0.1
nn_learningrate = 0.001
epsilonstart = 1
epsilonend = 1
nfeatures = 66

## calculated
xmin = 0.1*screen_width
xmax = 0.9*screen_width
ymin = 0.1*screen_height
ymax = 0.9*screen_height
xmin = 0.1*screen_width
xmax = 0.9*screen_width
ymin = 0.1*screen_height
ymax = 0.9*screen_height
deltax = (xmax-xmin)/nsquares
deltay = (ymax-ymin)/nsquares
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)





