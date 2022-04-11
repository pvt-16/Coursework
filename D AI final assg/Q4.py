import random
import numpy as np
import math

rows = 2
cols = 4
coffeeStates = 2    # without coffee = 0, with coffee = 1
numberStates = rows * cols * coffeeStates

# 0.8 probability in the intended direction, and 0.1 probability to the right or left of the intended direction,
robotProbabilities = [0.1, 0.8, 0.1]
dicActions = {'N':0, 'E':1, 'S':2, 'W':3}
# CHANGE TO - North, South, East and West is 0, 1, 2, and 3,
actions  = list(dicActions.keys()) # ['N', 'E', 'S', 'W']

dicActionsOrdered = {'N':0, 'S':1, 'E':2, 'W':3}
actionsOrdered = list(dicActionsOrdered.keys())

Qtable = np.zeros((numberStates,len(actions)))

# (x, y) on gridworld is mapped as (col, row) in 2D array
(startX, startY) = (1,0)    # the agent starts at (0,1)
(coffeeX, coffeeY) = (0,1)      # a cup of coffee (in position (1, 0))
(fireplaceX, fireplaceY) = (0,3)    # fire-place in the building at position (3, 0)
(officeX, officeY) = (1,3)      #  an office (in position (3, 1))

withoutCoffeeState = 0
#State = (Position, Coffee)
terminalPositions= [(fireplaceX, fireplaceY), (officeX, officeY)]
terminalStates= [(terminalPositions[0], 1-withoutCoffeeState), (terminalPositions[0], withoutCoffeeState), (terminalPositions[1], 1-withoutCoffeeState)]
impossibleStates = [((coffeeX, coffeeY), 0)]

epsilon = 0.1
discountFactor = 0.95
alpha = 0.1

def getQTableIndex( coffee, row, col) -> int:   #Maps the gridworld row, col values and coffee states to the row index of Q table
    return coffee * (rows * cols) + row * cols + col

def collectCoffeeRL(goal: int, fire: int, movement: int):
    # initialization  

    # reward allocation for terminal position
    fireplaceIndex = getQTableIndex(withoutCoffeeState,fireplaceX,fireplaceY)
    for col in range(len(Qtable[fireplaceIndex])):
        Qtable[fireplaceIndex][col] = fire
   
    fireplaceIndex = getQTableIndex(1-withoutCoffeeState,fireplaceX,fireplaceY)
    for col in range(len(Qtable[fireplaceIndex])):
        Qtable[fireplaceIndex][col] = fire

    officeIndex = getQTableIndex(1-withoutCoffeeState, officeX ,officeY)
    for col in range(len(Qtable[officeIndex])):
        Qtable[officeIndex][col] = goal

    startPosition = (startX, startY)
    startState = (startPosition, withoutCoffeeState)

    #TD Q-Learning algorithm
    iterCount = 0
    while (iterCount< 10001):        
        currentState = startState #( (startX, startY), withoutCoffeeState)
        while (currentState not in terminalStates):
            (position, coffeeState) = currentState
            (row, col) = position
            rowIndexQ = getQTableIndex(coffeeState, row, col)
            
            # update the Q values for coffee collection point in no Coffee state to get robot to move there
            noCoffeeCollectIndex = getQTableIndex(withoutCoffeeState, coffeeX ,coffeeY)
            coffeeCollectIndex = getQTableIndex(1-withoutCoffeeState, coffeeX ,coffeeY)
            for col in range(len(Qtable[noCoffeeCollectIndex])):
                Qtable[noCoffeeCollectIndex][col] = Qtable[coffeeCollectIndex][col]

            # using Epsilon - greedy approach
            randExplorationVal = random.random() # TRY - random.uniform(0, 1) 

            if (randExplorationVal < epsilon):
                # PLAY RANDOM ACTION
                randAction = random.random()
                # if randAction = 1, last action ('W') must be chosen explicitly else error
                # Follows North, South, East,  West 
                colIndexQ = math.floor(randAction*4) if randAction != 1 else len(actionsOrdered)-1
            else:
                #PLAY BEST ACTION
                stateQvalues = Qtable[rowIndexQ]
                indices = [i for i, x in enumerate(stateQvalues) if x == max(stateQvalues)]
                # Calling random choice even if there is only 1 machine to sync the random state value with expected o/p
                colIndexQ = random.choice(indices)  
            
            oldValue = Qtable[rowIndexQ, colIndexQ] # Q (s, a)
            selectedActionIndex= colIndexQ   # action index

            # TRAVEL TO NEXT STATE
            actualDirIndex = selectedActionIndex
            # Part 1 :Reaching next state depends on robot probabilities
            randMovement = random.random()
            if randMovement > 0.8 and randMovement < 0.9:   #Moves LEFT of directions
                actualDirIndex = selectedActionIndex-1
            elif randMovement >= 0.9:   #Moves RIGHT of directions
                actualDirIndex =  0 if selectedActionIndex +1 ==len(actions) else selectedActionIndex +1
            selectedAction = actions[actualDirIndex]
            
            # Part 1.1 update the position based on the direction
            (newX,newY) = position 
            # X index changes only while going north and south
            if selectedAction == 'N' and newX!= 0:  # north => row -1
                newX = newX-1
            elif selectedAction == 'S' and newX!= rows-1:  # south => row +1
                newX = newX+1
            # check intended Y - Y index changes only while going east and west
            elif selectedAction == 'E' and newY != cols-1:  # east => col +1
                newY = newY+1
            elif selectedAction== 'W' and newY!= 0:  # west => col -1
                newY = newY-1
            
            newPosition = (newX,newY)
            
            # Part 2: Next state also depends on if coffee was present or not
            if coffeeState == 1:   #Has cofee 
                randCoffee = random.random()
                if randCoffee >= 0.9:   # and Drops Coffee
                    coffeeState = 1- coffeeState

            newState = (newPosition, coffeeState) #new state is acheived - factors = has coffee, directional probability and epsilon-greedy approach
            newStateIndex =  getQTableIndex(coffeeState, newX, newY)  #  coffeeState * (rows * cols) + newX * cols + newY

            reward = movement
            if newPosition == (fireplaceX, fireplaceY):
                reward = fire
            elif newPosition == (officeX, officeY) and coffeeState == 1:# new position = office
                reward = goal
                    
            #get indices (actions) in the new State which have highest Q values 
            actionsIndexList = [i for i, x in enumerate(Qtable[newStateIndex]) if x == max(Qtable[newStateIndex])] 
            selectedDirIndex = random.choice(actionsIndexList) if len(actionsIndexList)> 1 else actionsIndexList[0]  # if multiple, random choice
            nextMaxQ = Qtable[newStateIndex, selectedDirIndex]

            #Q(s,a)= (1 - α) Q(s,a) + α(R(s) + λ max(Q(s', a')))
            newValue = (1 - alpha) * oldValue + alpha * (reward + discountFactor * nextMaxQ)

            # UPDATE THE OLD STATE VALUE
            Qtable[rowIndexQ, colIndexQ] = newValue    #Update Q(s,a) with new value
            currentState = newState
        iterCount+=1

    # FIND POLICIES - ONE MORE ITERATION
    policies = []
    for coffeeState in range(coffeeStates): #CoffeeStates = 0,1 =>0=no coffee, 1= has coffee
        rowPolicy=[]
        for row in range(rows):
            colPolicy = []
            for col in range(cols):
                position = (row, col)
                state = (position, coffeeState)
 
                if state in terminalStates or state in impossibleStates:
                    policyDir = '.'
                else:
                    rowIndex = getQTableIndex(coffeeState, row, col)
                    dirIndexList = [i for i, x in enumerate(Qtable[rowIndex]) if x == max(Qtable[rowIndex])] #get indices which have highest Q values
                    #dirIndexList = Qtable[rowIndex].indices  
                    policyDir = actions[random.choice(dirIndexList)]    #random selection of best ones

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

# REF
# https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

# MOODLE gives time exceeded error -
# output for 10000 runs is - 

# No Coffee:
# S . W .
# S W W S
# Coffee:
# E E W .
# S N E .
random.seed(0)
collectCoffeeRL(1,-1,0)