import random
class Arm:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def pull(self):
        returnValue =  random.gauss(self.mu, self.sigma)
        if (returnValue <= 0):
            returnValue = 0.0
        elif (returnValue >= 1.0):
            returnValue = 1.0
        return returnValue

import math
from typing import List
class UCB1:
    def __init__(self, arms: List[Arm]):
        self.arms = arms
        self.trackMachinePlay = [0]* len(self.arms) #to track number of times each machine is played
        self.trackMachineReward = [0]* len(self.arms)   #to track amount of reward won for each machine
        self.countPlays = 0     # to track number of plays

    def play(self):       
        selectedMachineIndex = 0 
        n_j = 0 
        if self.countPlays < len(self.arms):    # arms will be played in order when they are being played for the first time. 
            selectedMachineIndex = self.countPlays  #order is same as the number of plays
        else:
            expectedRewardsList = []
            for armIndex in range(len(self.arms)):
                n_j = self.trackMachinePlay[armIndex]   #number of times machine is played
                averageReward = self.trackMachineReward[armIndex] / self.trackMachinePlay[armIndex] #reward from playing this machine before
                if self.countPlays!=0 and n_j != 0:
                    val = averageReward + math.sqrt( (2 * math.log(self.countPlays))/ n_j)  #reward expectation calculation using formula in UCB1 algo
                    expectedRewardsList.append((val, armIndex))
            #print(expectedRewardsList)
            highestRewardMachine = sorted(expectedRewardsList, reverse=True)[0] #get the highest expected reward possible
            highestRewardsMachines = list(filter(lambda reward: reward[0] == highestRewardMachine[0], expectedRewardsList)) #machines delivering same reward

            # random selection if tie
            # MODIFIED = "Yes, in my own code I always call random.choice, even to break a tie of a single action"
            # Calling random choice even if there is only 1 machine to sync the random state value with expected o/p
            selectedMachine = random.choice(highestRewardsMachines)  
            selectedMachineIndex = selectedMachine[1]
            
        reward = self.arms[selectedMachineIndex].pull()     #play the machine
        self.trackMachineReward[selectedMachineIndex] += reward     #add the reward to selected machine
        self.trackMachinePlay[selectedMachineIndex] += 1
        self.countPlays += 1
        return selectedMachineIndex

random.seed(0)
arms = [None] * 5
# mu = [0.0001, 0.0000005, 0.000007, 0.0000008, 0.000001]
# sigma = [0.00001, 0.00002, 0.000001, 0.000001, 0.000002]

mu = [0.2, 0.5, 0.7, 0.8, 0.1]
sigma = [0.1, 0.2, 0.1, 0.1, 0.2]

for a in range(len(arms)):
    arms[a] = Arm(mu[a], sigma[a])
    print(arms[a])

ucb = UCB1(arms)
for trials in range(100):
    print(ucb.play(),end="")