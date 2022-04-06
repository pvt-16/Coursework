import random
from typing import Tuple
import math

epsilon = 0.1
discountFactor = 0.95
gridWorldCurrentUtilities = [[0] * 4, [0] * 4]
rows = len(gridWorldCurrentUtilities)
cols = len(gridWorldCurrentUtilities[0])
numberStates = rows * cols * 2
#North, South, East and West is 0, 1, 2, and 3,
#dicActions = {'N':0, 'S':1, 'E':2, 'W':3}
#arrRotation = [0, 2, 1, 3]  # North, East, South, West
robotProbabilities = [0.1, 0.8, 0.1]

# 0.8 probability in the intended direction, and 0.1 probability to the right or left of the intended direction,
dicActions = {'N':0, 'E':1, 'S':2, 'W':3}

class MDP:
    def __init__(self, states, actions, transitionFunc, rewards, terminalStates) -> None:
        return

def getQvalue (intendedDir,  position: tuple()):
    actions  = list(dicActions.keys()) # ['N', 'E', 'S', 'W']
    intendedDirIndex = dicActions[intendedDir]
    selectedDirections = [actions[intendedDirIndex-1],actions[intendedDirIndex], actions[0 if intendedDirIndex +1 ==len(dicActions) else intendedDirIndex +1]]
    # FORMAT of selected directions [left of intended, intended, right of intended]
    hasCoffeeProb = 0.9
    tempSum = 0
    # Transition probabilites and rewards depend on current state only
    (newX,newY) = position 
    for idxDir in range(len(selectedDirections)):
        # X index changes only while going north and south
        if selectedDirections[idxDir] == 'N':  # north => row -1
            if newX!= 0:
                newX = newX-1
        elif selectedDirections[idxDir] == 'S':  # south => row +1
            if newX!= rows-1:
                newX = newX+1

        # check intended Y - Y index changes only while going east and west
        elif selectedDirections[idxDir] == 'E':  # east => col +1
            if newY != cols-1:
                newY = newY+1
        elif selectedDirections[idxDir] == 'W':  # west => col -1
            if newY!= 0:
                newY = newY-1
        
        tempSum += robotProbabilities[idxDir] * gridWorldCurrentUtilities[newX][newY]       #sigma P(s0 |s, a) ∗ U(s0)
    # The randomness in the actual movement given an action,     
    # and the randomness on dropping the coffee or not, define the transition function of the MDP model.
    return tempSum

def collectCoffee(goal: int, fire: int, movement: int):
    # initialization
    hasCoffee: bool = False

    # (x, y) on gridworld is mapped as (col, row) in 2D array
    (currentX, currentY) = (1,0)    # (starting at position (0, 1))
    (coffeeX, coffeeY) = (0,1)      # a cup of coffee (in position (1; 0))
    (fireplaceX, fireplaceY) = (0,3)    # fire-place in the building at position (3, 0)
    (officeX, officeY) = (1,3)      #  an office (in position (3, 1))
    
    #State = (Position, Coffee)
    terminalPositions= [(fireplaceX, fireplaceY), (officeX, officeY)]
    terminalStates= [((fireplaceX, fireplaceY), True ), ((fireplaceX, fireplaceY), False ), (officeX, officeY, True)]

    #rewards allocation
    gridWorldCurrentUtilities[currentX][currentY] = 0     
    gridWorldCurrentUtilities[fireplaceX][fireplaceY] = fire  
    gridWorldCurrentUtilities[officeX][officeY] = goal  

    #VALUE ITERATION ALGORITHM
    converged: bool = False
    while (converged == False):
        utilityPrime = movement
        differencesList = []
        for row in range(len(gridWorldCurrentUtilities)):
            for col in range(len(gridWorldCurrentUtilities[0])):
                position = (row, col)
                state = (position, hasCoffee)
                utility = gridWorldCurrentUtilities[row][col]
                # U0(s) = R(s) +  discountfactor *  max( sigma P(s0 |s, a) ∗ U(s0))
                #newUtility = movement + max( sum of getQvalue(   probability * utility ))
                tempQvalues = []
                if position in terminalPositions:
                    continue
                for direction in dicActions:
                    tempQvalue = getQvalue(direction, state)  # get Q-values for all directions
                    tempQvalue = movement + discountFactor * tempQvalue     #  R(s) +  discountfactor * Qvalue
                    tempQvalues.append(tempQvalue)
                
                #get max
                maxQvalue = sorted(tempQvalues, reverse=True)[0]
                utilityPrime =  maxQvalue   # max(q values)
                gridWorldCurrentUtilities[row][col]  = utilityPrime     # Assign U0(s)
                difference = abs(utilityPrime - utility)    
                differencesList.append(difference)      #store differences for all states
        
        maxDifference = sorted(differencesList, reverse= True)[0]   #get maximum difference
        if (maxDifference < epsilon):
            converged = True
    # For policy calculation
    if hasCoffee:
        targetPosition = gridWorldCurrentUtilities[officeX][officeY]
    else:
        targetPosition = gridWorldCurrentUtilities[coffeeX][coffeeY]    

# When you are calculating the Bellman's equation, you need to 
# sum across all possible next states, 
# and multiply their utility with their probability.
# Given a state with coffee, the agent may or may not have coffee in the next state, 
# and you have to consider that in the Bellman equation.

    print("No Coffee:")


    print("Coffee:")
    return

random.seed(0)
collectCoffee(1,-1,0)

# No Coffee:
# E . W . 
# N N W W 
# Coffee:
# E S W . 
# E E E .