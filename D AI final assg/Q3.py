import random

def collectCoffee(goal: int, fire: int, movement: int):
    # initialization
    discountFactor = 0.95
    epsilon = 0.001
    gridWorld = [[0] * 4, [0] * 4]

    gridWorld[0][3] = -1
    gridWorld[1][3] = 1
    return

random.seed(0)
collectCoffee(1,-1,0)