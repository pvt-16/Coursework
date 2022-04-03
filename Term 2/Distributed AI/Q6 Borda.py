import random
from typing import List

def bordaAggregation(opinions: List[list]):
    number_nn = len(opinions)   # number of neural networks
    number_labels = len(opinions[0])    # number of labels
    sum_of_opinions: list = [0]* number_labels  
    #STEP 1: Find sum of probabilities
    for l in range(number_labels):     #loop for sum of probabilities for each label
        for n in range(number_nn):
            sum_of_opinions[l] += opinions[n][l]
    
    #STEP 2: Rank order the labels
    rank_order: list = [-1]* number_labels
    unique_votes = list(set(sum_of_opinions))   #find unique probabilities
    unique_votes.sort(reverse=True)     #sort probabilities in descending order to get highest-lowest ranking
    rank_counter = 0    #counter to assign rank
    loop_counter = 0
    ranking: list = [-1]* number_labels
    for uv in unique_votes:
        n_instances = sum_of_opinions.count(uv)     #check for multiple 
        if n_instances==1:
            selected_index = sum_of_opinions.index(uv)  #get the index and assign rank
            ranking[loop_counter] = selected_index
            #rank_order[selected_index] = rank_counter  
            #rank_counter = rank_counter+1   #increment rank counter
        else:   #multiple labels have same probability
           all_indexes_votes = [i for i in range(len(sum_of_opinions)) if sum_of_opinions[i] == uv] #get indices for same values
           for elem in range(n_instances):
               selected_index = random.choice(all_indexes_votes) #choose random index for ranking
               ranking[elem + loop_counter] = selected_index
               #rank_order[selected_index] = rank_counter   #assign rank to element
               #rank_counter = rank_counter+1    
               all_indexes_votes.remove(selected_index)  #to avoid replacement during random choice
        
        loop_counter= loop_counter+ n_instances
        
        #rank_counter = rank_counter+1

    # for i in range(number_labels):
    #     ranking[i] = rank_order.index(i)

    return ranking


random.seed(0)
agent0 = [0.25,0.2,0.15,0.2,0.2]
agent1 = [0.25,0.2,0.15,0.2,0.2]
agent2 = [0.25,0.2,0.15,0.2,0.2]
print(bordaAggregation([agent0,agent1,agent2]))