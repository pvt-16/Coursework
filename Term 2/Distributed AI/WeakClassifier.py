# A fake Weak Classifier for SCC462 Assignment.
# Leandro Soriano Marcolino

from cmath import log
import random
from typing import List
import math
from sklearn.feature_selection import SelectFdr

from sympy import beta

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

class AdaBoost():
    def __init__(self):
        self.weights = []
        self.betas = 0
    def train(self, dataset, labels, D, T):
        listOfClassifiers = [WeakClassifier() for i in range(T)] #maintain list of T classifiers - one for each iteration
        list_betas = []
        self.weights = D   # setting initial weights
        for t in range(T):   #iterating T times 
            error_t = 0
            p_t = [weight_i / sum(self.weights) for weight_i in self.weights]  #calculating normalized weight distribution
            current_classifier: WeakClassifier =  listOfClassifiers[t]  
            
            #AdaBoost Training returns error and the difference between predicted values and actual values
            error_t, hypothesis_diff = self._AdaBoostTraining( dataset, labels, current_classifier, p_t)
            if (error_t >= 0.5): #if error is more than 50%, we need to re-train it
                while(error_t >= 0.5):
                    error_t, hypothesis_diff = self._AdaBoostTraining( dataset, labels, current_classifier, p_t)
            
            beta_t = error_t / (1-error_t) #calculate importance of classifier
            list_betas.append(beta_t)
            #recalculate the weights 
            self.weights = [self.weights[i]*(math.pow(beta_t,(1 - hypothesis_diff[i]))) for i in range(len(self.weights))]

        self.betas = list_betas
        self.classifiers = listOfClassifiers

    def classify(self, item):
        sum_log_beta_t = sum(math.log(1/beta_t) for beta_t in self.betas) #AdaBoost algo for 
        # #AdaBoost algo for sum of hypotheses
        sum_hypotheses = sum(math.log(1/self.betas[t]) * self.classifiers[t].classify(item) for t in range(len(self.betas)))
        if sum_hypotheses > 0.5 * sum_log_beta_t:       #return hypothesis 
            return 1
        else:
            return 0

    def _AdaBoostTraining(self, dataset, labels, current_classifier, p_t): #private method to train AdaBoost
        current_classifier.train(dataset, labels, p_t) #WeakClassifier training
        hypothesis_t = [current_classifier.classify(item) for item in range(len(dataset))] #Predictions from WeakClassifier
        hypothesis_diff = [abs(hypothesis_t[i]- labels[i]) for i in range(len(dataset))]    #Difference in predictions and actuals
        error_t = sum([(p_t[i] * hypothesis_diff[i]) for i in range(len(dataset))])     # SUm of weighted differences
        return (error_t, hypothesis_diff)

random.seed(0)
labels = [0, 1, 1, 1, 1, 0]
dataset = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]]
D = [1/6]*6
adaBoost = AdaBoost()
adaBoost.train(dataset,labels,D,3)

for n in range(len(dataset)):
    print(adaBoost.classify(n), end=" ") #adaBoost.classify(n)

#0 0 1 0 1 0