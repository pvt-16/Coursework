# A fake Weak Classifier for SCC462 Assignment.
# Leandro Soriano Marcolino

import random

class WeakClassifier:

    def __init__(self):
        self._predictedLabels = []
    
    def train(self, dataset, labels, p):
        # We are simulating a training process here, this is not a real classifier.       
        self._predictedLabels = [0]*len(labels)

        for n in range(len(labels)):
            r = random.random()

            # Let's pretend odd items are harder
            if (n % 2 == 0):
                correctProb = 0.6
            else:
                correctProb = 0.55
            
            if (r < correctProb):
                self._predictedLabels[n] = labels[n]
            else:
                self._predictedLabels[n] = (1 - labels[n])

    def classify(self, item):        
        return self._predictedLabels[item]

class AdaBoost(WeakClassifier):
    def __init__(self):
        super().__init__()
    def train(self, dataset, labels, D, T):
        return
    def classify(self, item):
        return super().classify(item)

random.seed(0)
labels = [0, 1, 1, 1, 1, 0]
dataset = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]]
D = [1/6]*6
adaBoost = AdaBoost()
adaBoost.train(dataset,labels,D,3)

for n in range(len(dataset)):
    print(adaBoost.classify(n), end=" ")

#0 0 1 0 1 0