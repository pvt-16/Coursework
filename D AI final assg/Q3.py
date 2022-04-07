import random

epsilon = 0.001
discountFactor = 0.95
gridWorldCurrentUtilities = [[[0] * 4, [0] * 4], [[0] * 4, [0] * 4]] 
rows = len(gridWorldCurrentUtilities[0])
cols = len(gridWorldCurrentUtilities[0][0])
coffeeStates = 2
numberStates = rows * cols * coffeeStates
# CHANGE TO - North, South, East and West is 0, 1, 2, and 3,

# (x, y) on gridworld is mapped as (col, row) in 2D array
(coffeeX, coffeeY) = (0,1)      # a cup of coffee (in position (1; 0))
(fireplaceX, fireplaceY) = (0,3)    # fire-place in the building at position (3, 0)
(officeX, officeY) = (1,3)      #  an office (in position (3, 1))

# 0.8 probability in the intended direction, and 0.1 probability to the right or left of the intended direction,
robotProbabilities = [0.1, 0.8, 0.1]
dicActions = {'N':0, 'E':1, 'S':2, 'W':3}

def getQvalue (intendedDir,  position: tuple(), coffeeState: int):
    actions  = list(dicActions.keys()) # ['N', 'E', 'S', 'W']
    intendedDirIndex = dicActions[intendedDir]
    # FORMAT of selected directions [left of intended, intended, right of intended]
    selectedDirections = [actions[intendedDirIndex-1],actions[intendedDirIndex], actions[0 if intendedDirIndex +1 ==len(dicActions) else intendedDirIndex +1]]
    
    hasCoffeeProb = 0.9 #ROBOT can drop the coffee 0.1 probability
    tempSum = 0
    # In no coffee scenario, the target is the coffee table, not office 
    # The utility of that position in no coffee state is the same as the utility of that position with coffee  
    gridWorldCurrentUtilities[0][coffeeX][coffeeY] = gridWorldCurrentUtilities[1][coffeeX][coffeeY] 
    # Transition probabilites and rewards depend on current state only
    for idxDir in range(len(selectedDirections)):
        (newX,newY) = position 
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
            tempSum += robotProbabilities[idxDir] * gridWorldCurrentUtilities[coffeeState][newX][newY] * hasCoffeeProb #sigma P(s0 |s, a) ∗ U(s0) with cofee
            tempSum += robotProbabilities[idxDir] * gridWorldCurrentUtilities[1-coffeeState][newX][newY]* (1-hasCoffeeProb)
        else:   #coffeeState=0 => doesn't have cofee, can't drop it, goes to new state without coffee
            tempSum += robotProbabilities[idxDir] * gridWorldCurrentUtilities[coffeeState][newX][newY] #sigma P(s0 |s, a) ∗ U(s0)
    return tempSum

def collectCoffee(goal: int, fire: int, movement: int):
    # initialization  
    #State = (Position, Coffee)
    terminalPositions= [(fireplaceX, fireplaceY), (officeX, officeY)]
    terminalStates= [(terminalPositions[0], 1), (terminalPositions[0], 0), (terminalPositions[1], 1), ((coffeeX, coffeeY), 0)]

    # reward allocation for terminal position
    # gridWorldCurrentUtilities[officeX][officeY][0] = 0
    gridWorldCurrentUtilities[0][coffeeX][coffeeY] = gridWorldCurrentUtilities[1][coffeeX][coffeeY] 
    gridWorldCurrentUtilities[0][fireplaceX][fireplaceY]= fire
    gridWorldCurrentUtilities[1][fireplaceX][fireplaceY]= fire  
    gridWorldCurrentUtilities[1][officeX][officeY] = goal  

    #VALUE ITERATION ALGORITHM
    converged: bool = False
    while (converged == False):
        utilityPrime = movement
        differencesList = []
        for coffeeState in range(coffeeStates): #CoffeeStates = 0,1 =>0=no coffee, 1= has coffee
            for row in range(len(gridWorldCurrentUtilities[0])):
                for col in range(len(gridWorldCurrentUtilities[0][0])):
                    position = (row, col)
                    state = (position, coffeeState)
                    utility = gridWorldCurrentUtilities[coffeeState][row][col]
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
                    gridWorldCurrentUtilities[coffeeState][row][col] = utilityPrime     # Assign U0(s)
                    difference = abs(utilityPrime - utility)    
                    differencesList.append(difference)      #store differences for all states
        
        maxDifference = sorted(differencesList, reverse= True)[0]   #get maximum difference
        if (maxDifference < epsilon):
            converged = True

    # FIND POLICIES - ONE MORE ITERATION
    policies = []
    for coffeeState in range(coffeeStates): #CoffeeStates = 0,1 =>0=no coffee, 1= has coffee
        rowPolicy=[]
        for row in range(rows):
            colPolicy = []
            for col in range(cols):
                position = (row, col)
                state = (position, coffeeState)
                utility = gridWorldCurrentUtilities[coffeeState][row][col]
                # U0(s) = R(s) +  discountfactor *  max( sigma P(s0 |s, a) ∗ U(s0))
                #newUtility = movement + max( sum of getQvalue(  probability * utility ))
                tempQvalues = []
                if state in terminalStates:
                    policyDir = '.'
                else:
                    for direction in dicActions:
                        tempQvalue = getQvalue(direction, position, coffeeState)  # get Q-values for all directions
                        tempQvalue = movement + discountFactor * tempQvalue     #  R(s) +  discountfactor * Qvalue
                        tempQvalues.append((tempQvalue, direction))
                        #get max
                    maxQvalue = sorted(tempQvalues, reverse=True)[0]
                    (utilityPrime, policyDir) =  maxQvalue   # max(q values)
                    
                colPolicy.append(policyDir)
            rowPolicy.append(colPolicy)
        policies.append(rowPolicy)

    print("No Coffee:")
    for row in  range(rows):
        for col in range(cols):
            print (policies[0][row][col], end=" ")
        print("")

    print("Coffee:")
    for row in  range(rows):
        for col in range(cols):
            print(policies[1][row][col], end=" ")
        print("")

# random.seed(0)
# collectCoffee(1,-1,0)

# No Coffee:
# E . W . 
# N N W W 
# Coffee:
# E S W . 
# E E E .