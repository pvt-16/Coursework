import random

discountFactor = 0.95
epsilon = 0.001
gridWorld = [[0] * 4, [0] * 4]

#North, South, East and West is 0, 1, 2, and 3,
dicActions = {'North':0, 'South':1, 'East':2, 'West':3}
arrRotation = [0, 2, 1, 3]


def transitionFunc (intendedDir):
    robotProbabilities = [0.8, 0.1, 0.1]
    # Transition probabilites and rewards depend on current state only

    
    
    return

def collectCoffee(goal: int, fire: int, movement: int):
    # initialization
    # (x, y) on gridworld is mapped as (col, row) in 2D array

    gridWorld[1][0] = 0     # (starting at position (0, 1))
    gridWorld[0][3] = fire  # fire-place in the building at position (3, 0)
    gridWorld[1][3] = goal  #  an office (in position (3, 1))

    # a cup of coffee (in position (1; 0))

    return

random.seed(0)
collectCoffee(1,-1,0)