import random
from typing import Tuple
import numpy as np

epsilon = 0.1
discountFactor = 0.95
gridWorldCurrentUtilities = np.zeros((2,4,2)) # [[[0] * 4, [0] * 4], [[0] * 4, [0] * 4]] #

rows = len(gridWorldCurrentUtilities)
cols = len(gridWorldCurrentUtilities[0])
coffeeStates = 2
numberStates = rows * cols * coffeeStates
#North, South, East and West is 0, 1, 2, and 3,
#dicActions = {'N':0, 'S':1, 'E':2, 'W':3}
#arrRotation = [0, 2, 1, 3]  # North, East, South, West
robotProbabilities = [0.1, 0.8, 0.1]

# 0.8 probability in the intended direction, and 0.1 probability to the right or left of the intended direction,
dicActions = {'N':0, 'E':1, 'S':2, 'W':3}

# class MDP:
#     def __init__(self, states, actions, transitionFunc, rewards, terminalStates) -> None:
#         return

def getQvalue (intendedDir,  position: tuple(), coffeeState: int):
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
        if selectedDirections[idxDir] == 'N' and newX!= 0:  # north => row -1
            newX = newX-1
        elif selectedDirections[idxDir] == 'S' and newX!= rows-1:  # south => row +1
            newX = newX+1
        # check intended Y - Y index changes only while going east and west
        elif selectedDirections[idxDir] == 'E' and newY != cols-1:  # east => col +1
            newY = newY+1
        elif selectedDirections[idxDir] == 'W' and newY!= 0:  # west => col -1
            newY = newY-1
        
        if coffeeState == 1:
            # Given a state with coffee, the agent may or may not have coffee in the next state, consider that in the Bellman equation.
            tempSum += robotProbabilities[idxDir] * gridWorldCurrentUtilities[newX][newY][coffeeState] * hasCoffeeProb #sigma P(s0 |s, a) ∗ U(s0)
            tempSum += robotProbabilities[idxDir] * gridWorldCurrentUtilities[newX][newY][1-coffeeState] * (1-hasCoffeeProb)
        else:   #coffeeState=0 => doesn't have cofee, can't drop it, goes to new state without coffee
            tempSum += robotProbabilities[idxDir] * gridWorldCurrentUtilities[newX][newY][coffeeState] #sigma P(s0 |s, a) ∗ U(s0)
    return tempSum

def collectCoffee(goal: int, fire: int, movement: int):
    # initialization
    hasCoffee: bool = False

    # (x, y) on gridworld is mapped as (col, row) in 2D array
#    (currentX, currentY) = (1,0)    # (starting at position (0, 1))
#    (coffeeX, coffeeY) = (0,1)      # a cup of coffee (in position (1; 0))
    (fireplaceX, fireplaceY) = (0,3)    # fire-place in the building at position (3, 0)
    (officeX, officeY) = (1,3)      #  an office (in position (3, 1))
    
    #State = (Position, Coffee)
    terminalPositions= [(fireplaceX, fireplaceY), (officeX, officeY)]
    terminalStates= [(terminalPositions[0], True ), (terminalPositions[0], False), (terminalPositions[1], True)]

    #reward allocation for terminal position
#    gridWorldCurrentUtilities[officeX][officeY][0] = 0
    gridWorldCurrentUtilities[officeX][officeY][0] = goal
    gridWorldCurrentUtilities[fireplaceX][fireplaceY][0]= fire
    gridWorldCurrentUtilities[fireplaceX][fireplaceY][1]= fire  
    gridWorldCurrentUtilities[officeX][officeY][1] = goal  

    #VALUE ITERATION ALGORITHM
    converged: bool = False
    while (converged == False):
        utilityPrime = movement
        differencesList = []
        for row in range(len(gridWorldCurrentUtilities)):
            for col in range(len(gridWorldCurrentUtilities[0])):
                position = (row, col)
                for coffeeState in range(coffeeStates): #CoffeeStates = 0,1 =>0=no coffee, 1= has coffee
                    state = (position, coffeeState)
                    utility = gridWorldCurrentUtilities[row][col][coffeeState]
                    newUtility = 0
                    # U0(s) = R(s) +  discountfactor *  max( sigma P(s0 |s, a) ∗ U(s0))
                    #newUtility = movement + max( sum of getQvalue(  probability * utility ))
                    tempQvalues = []
                    if state in terminalStates:
                        continue
                    for direction in dicActions:
                        tempQvalue = getQvalue(direction, position, coffeeState)  # get Q-values for all directions
                        newUtility = movement + discountFactor * tempQvalue     #  R(s) +  discountfactor * Qvalue
                        tempQvalues.append(newUtility)
                    
                    #get max
                    maxQvalue = sorted(tempQvalues, reverse=True)[0]
                    utilityPrime =  maxQvalue   # max(q values)
                    gridWorldCurrentUtilities[row][col][coffeeState]  = utilityPrime     # Assign U0(s)
                    difference = abs(utilityPrime - utility)    
                    differencesList.append(difference)      #store differences for all states
        
        maxDifference = sorted(differencesList, reverse= True)[0]   #get maximum difference
        if (maxDifference < epsilon):
            converged = True
    
    #POLICY
    # For policy calculation
    # if hasCoffee:
    #     targetPosition = gridWorldCurrentUtilities[officeX][officeY]
    # else:
    #     targetPosition = gridWorldCurrentUtilities[coffeeX][coffeeY]    
    policies = []

    for coffeeState in range(coffeeStates): #CoffeeStates = 0,1 =>0=no coffee, 1= has coffee
        rowPolicy=[]
        for row in range(len(gridWorldCurrentUtilities)):
            colPolicy = []
            for col in range(len(gridWorldCurrentUtilities[0])):
                position = (row, col)
                state = (position, coffeeState)
                utility = gridWorldCurrentUtilities[row][col][coffeeState]
                # U0(s) = R(s) +  discountfactor *  max( sigma P(s0 |s, a) ∗ U(s0))
                #newUtility = movement + max( sum of getQvalue(  probability * utility ))
                tempQvalues = []
                if state in terminalStates:
                    policies[row][col][coffeeState] = '.'
                else:
                    for direction in dicActions:
                        tempQvalue = getQvalue(direction, position, coffeeState)  # get Q-values for all directions
                        tempQvalue = movement + discountFactor * tempQvalue     #  R(s) +  discountfactor * Qvalue
                        tempQvalues.append((tempQvalue, direction))
                    maxQvalue = sorted(tempQvalues, reverse=True)[0]
                    (utilityPrime, policyDir) =  maxQvalue   # max(q values)
                    
                #get max
                colPolicy.append(policyDir)
                #gridWorldCurrentUtilities[row][col][coffeeState]  = utilityPrime     # Assign U0(s)
                #difference = abs(utilityPrime - utility)    
                #differencesList.append(difference)      #store differences for all states
            rowPolicy.append(colPolicy)
        policies.append(rowPolicy)


    print("No Coffee:")
    for row in  range(len(policies)):
        for col in range(len(policies[0])):
            print (policies[row][col][0], " ")
        print("/n")

    print("Coffee:")
    for row in  range(len(policies)):
        for col in range(len(policies[0])):
            print (policies[row][col][1], " ")
        print("/n")
    return

random.seed(0)
collectCoffee(1,-1,0)

# No Coffee:
# E . W . 
# N N W W 
# Coffee:
# E S W . 
# E E E .