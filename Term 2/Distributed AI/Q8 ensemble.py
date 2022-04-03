# Simple Logistic Regression classifier for SCC462 Assignment. Adapted from solution of SCC461 CW 10.
# Leandro Soriano Marcolino

from audioop import avg
from itertools import count
from operator import index
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


def predictionFeatureVector(batch: List[list]) -> list: #prediction for one batch
    #list = [c0_vote, c1_vote, c2_vote, decision, true_label] 
    featureVector: list = [0]*6 #  {c0}, {c1}, {c2}, {c0; c1}, {c0; c2}, {c1; c2}.
    for vote_list in batch:
        local_vote_list = vote_list.copy()
        true_label = local_vote_list.pop()
        decision = local_vote_list.pop()
        if (len(set(local_vote_list)) == len(local_vote_list)): #all classifiers were different, set gives unique classifier numbers
            classifier_correct=local_vote_list.index(decision) # find one that was randomly selected
            featureVector[classifier_correct] += 1 #increase count for that one classifier
        elif len(set(local_vote_list)) != 1:  #majority voting took place
            #find classifier combinations with same vote and final vote
            list_of_indices = []
            for i in  range(len(local_vote_list)):
                if local_vote_list[i] == decision: #if the classifier vote counts towards final decision
                    list_of_indices.append(i)   #add it to the list 
            sum_indices = sum(list_of_indices)
            combo_position = len(local_vote_list)-1 + sum_indices   #find position of classifier combo in feature vector
            featureVector[combo_position] += 1
        
        #else all classifiers gave same o/p, no one made the decision, so we don't add   

    featureVector = [featureVector_i/ len(batch) for featureVector_i in featureVector]
    return featureVector

def successPrediction(batches: List[List[list]]) -> LogisticRegression:
    feature_vector_dataset = []; feature_vector_labels = []
    for batch in batches:
        feature_vector_dataset.append(predictionFeatureVector(batch))   #we train on feature vector of batches
        count_correct_decisions = 0
        for vote_list in batch:
            local_vote_list = vote_list.copy()
            true_label = local_vote_list.pop()  #get ground truth label
            decision = local_vote_list.pop()    # get ensemble decision
            count_correct_decisions += 1 if true_label == decision else 0
        label = count_correct_decisions/ len(batch)     #creating output set to train Logistic regression model
        feature_vector_labels.append(label)
        
    #classifier works on 6 features from feature Vector 
    classifier: LogisticRegression = LogisticRegression(nFeatures=len(feature_vector_dataset[0]), threshold = sum(feature_vector_labels)/ len(feature_vector_labels))
    classifier.train(feature_vector_dataset, feature_vector_labels)  #threshold is the mean of correct answers in each batch 
    #print (feature_vector_dataset, feature_vector_labels)
    return classifier

#Test casses
## TC 1
batch0 = [[0,1,0,0,1],[1,1,0,1,1],[0,1,2,0,0],[0,0,0,0,0]]
print(predictionFeatureVector(batch0))
# [0.25, 0.0, 0.0, 0.25, 0.25, 0.0]

## TC 2
batch1 = [[0,0,0,0,0],[1,1,2,1,2],[0,1,1,1,1],[2,2,1,2,2]]
print(predictionFeatureVector(batch1))
# # [0.0, 0.0, 0.0, 0.5, 0.0, 0.25]

# # TC 3
batch2 = [[0,1,2,1,2],[0,1,2,1,1],[0,1,2,0,2],[0,2,2,2,2]]
print(predictionFeatureVector(batch2))
# # [0.25, 0.5, 0.0, 0.0, 0.0, 0.25]

# # TC 4
batch3 = [[0,0,0,0,1],[2,2,1,2,2],[1,2,0,2,0],[2,1,0,1,1]]
print(predictionFeatureVector(batch3))
# #[0.0, 0.5, 0.0, 0.25, 0.0, 0.0]

random.seed(0)
batches = [batch0, batch1, batch2, batch3]
classifier = successPrediction(batches)
for n in range(len(batches)):
    print(classifier.classify(predictionFeatureVector(batches[n])),end=" ")

#1 1 0 0 