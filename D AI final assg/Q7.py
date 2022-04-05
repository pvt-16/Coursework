import random
import math
from typing import List

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

class UCB1:
    def __init__(self, arms: List[Arm]):
        self.arms = arms

    def play():
        return

random.seed(0)
arms = [None] * 5
mu = [0.2, 0.5, 0.7, 0.8, 0.1]
sigma = [0.1, 0.2, 0.1, 0.1, 0.2]
for a in range(len(arms)):
    arms[a] = Arm(mu[a], sigma[a])
    print(arms[a])

# ucb = UCB1(arms)
# for trials in range(100):
#     print(ucb.play(),end="")