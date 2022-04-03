# Simple Logistic Regression classifier for SCC462 Assignment. Adapted from solution of SCC461 CW 10.
# Leandro Soriano Marcolino

from itertools import count
import random
import math
from typing import List

class LogisticRegression:
    # Initialise learning rate (alpha), threshold and weights
    def __init__(self, nFeatures, alpha = 0.15, threshold = 0.5, nEpochs = 200):
        self._alpha = alpha
        self._nFeatures = nFeatures
        self._weights = []
        self._threshold = threshold
        self._nEpochs = nEpochs
        
    # Calculates the output of the logistic regression. Note that we do not round here.
    def _logistic(self, item):
        t = 0
        for w in range(len(self._weights)): # We use a for loop for the linear combination, to be more general to any number of features
            t += item[w]*self._weights[w]
        return 1/(1 + math.exp(-t)) # Math.exp is the natural exponential function (e^x)
        
    def train(self, dataset, labels):
        # Initialising weights
        self._weights = [random.random()]*(self._nFeatures+1)
        
        # Pre-initialising the lists
        dataWithLabels = [0]*len(dataset)
        MSEList = [0]*self._nEpochs
            
        # We combine the dataset with the labels, so that we can shuffle without losing the label information
        for i in range(len(dataset)):
            dataWithLabels[i] = dataset[i] + [labels[i]]

        # Shuffle changes the list, so we make a copy of it first
        newData = dataWithLabels.copy()

        for epoch in range(self._nEpochs):
            # The dataset is randomly shuffled at the beginning of each epoch
            random.shuffle(newData)

            #import ipdb; ipdb.set_trace()
            
            # For each epoch, we need to keep track of the MSE
            error = 0
            
            for item in newData:
                item = [1] + item # We add a 1 to the beginning of each item, to account for the independent term
                sigma = self._logistic(item[:-1]) # Now item also has the label in the end, so need to adjust when passing to the logistic function

                # Similarly, here we skip the last column of the item, since it is the label
                for w in range(len(item)-1):
                    self._weights[w] -= self._alpha * (sigma - item[-1]) * (sigma) * (1 - sigma) * item[w]

                # Update the sum of squared errors
                error += (sigma - item[-1])**2

            MSEList[epoch] = error/len(newData)

        return MSEList

    def classify(self, item):
        item = [1] + item

        sigma = self._logistic(item)

        if (sigma <= self._threshold):
            return 0
        else:
            return 1


class StackedGeneralisation():
    def __init__(self,classifiers: List[LogisticRegression],aggregator: LogisticRegression):
        self.classifiers = classifiers      #Tier-1 classifiers
        self.aggregator = aggregator    #Tier-2 meta classifier
    
    def train(self, dataset:List[list], labels: List[int]):
        tier_2_classifer_dataset = []
        tier_2_classifier_labels = []
        length_dataset = len(dataset)-1
        counter = 0 #length_dataset
        for n in range(len(dataset)):
            #STEP 1: Training all classifiers
            local_dataset = dataset.copy()  # Copy dataset to modify later     
            local_labels = labels.copy()    #Copy labels to modify later
            cross_validation_data = local_dataset.pop(counter)  #remove one batch for cross validation
            cross_validation_label = local_labels.pop(counter)  
            predicted_labels_all = []
            for current_classifer in self.classifiers:
                current_classifer.train(dataset= local_dataset, labels= local_labels)   #train on remaining data
                #STEP 2: Predictions
                predicted_label = current_classifer.classify(cross_validation_data)     #get decisions for trained classifier
                predicted_labels_all.append(predicted_label)
            
            counter = 0 if (counter >= length_dataset) else counter+1
            # Creating the dataset from Tier-1 decisions for Meta classifier
            tier_2_classifer_dataset.append(predicted_labels_all)   #Predicted values from all the classifiers for validation dataset
            tier_2_classifier_labels.append(cross_validation_label)     #Expected output for training data in validation dataset
        
        # STEP 3: Training meta classifier
        self.aggregator.train(tier_2_classifer_dataset, tier_2_classifier_labels)
        #STEP 4: retrain all classifiers
        for current_classifer in self.classifiers:
            current_classifer.train(dataset= dataset, labels= labels)
        return

    def classify(self, item: list): 
        tier_1_predictions = [] #Store predictions from Tier-1 Classifiers 
        for current_classifer in self.classifiers:
            tier_1_predictions.append(current_classifer.classify(item)) #Get all predictions

        sigma = self.aggregator.classify(tier_1_predictions) #get weighted sum of Tier-1 predictions
        return 0 if (sigma <= self.aggregator._threshold) else 1   #check applied threshold for classification

random.seed(0)
dataset = [[0.2,0.4],[0.3,0.7],[0.9,0.7],[0.8,0.9], [0.8,0.9]]
labels = [0, 0, 1, 1, 1]

classifiers = []
classifiers.append(LogisticRegression(2))
classifiers.append(LogisticRegression(2,0.3))
classifiers.append(LogisticRegression(2,0.3,0.7))
classifiers.append(LogisticRegression(2,0.01))
classifiers.append(LogisticRegression(2,0.01,0.3))

aggregator = LogisticRegression(5)

st = StackedGeneralisation(classifiers, aggregator)

st.train(dataset, labels)
for n in range(len(dataset)):
    print(st.classify(dataset[n]), end = " ")

# 0 0 1 1 