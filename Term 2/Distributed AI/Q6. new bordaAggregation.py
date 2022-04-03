import random
from typing import List

def bordaAggregation(opinions: List[list]):
    number_nn = len(opinions)   # number of neural networks
    number_labels = len(opinions[0])    # number of labels
    sum_of_opinions: list = [0]* number_labels  
    ranked_opinions:List[list] = []
    #STEP 1: Get ranked opinions
    for n in range(number_nn):
        unique_votes = list(set(opinions[n]))
        unique_votes.sort(reverse=True)
        vote_rank = number_labels
        ranked_opinion_list = [0]* number_labels
        for uv in unique_votes:
            n_instances = opinions[n].count(uv)     #check for multiple 
            if n_instances==1:
                selected_index = opinions[n].index(uv)
                ranked_opinion_list[selected_index] = opinions[n][selected_index] * vote_rank
                vote_rank = vote_rank-1
            else:
                all_indexes_votes = [i for i in range(len(opinions[n])) if opinions[n][i] == uv] #get indices for same values
                for j in range(n_instances):
                    selected_index = random.choice(all_indexes_votes)
                    ranked_opinion_list[selected_index] = opinions[n][selected_index] * vote_rank
                    all_indexes_votes.remove(selected_index)
                    vote_rank = vote_rank-1
        ranked_opinions.append(ranked_opinion_list)

    #STEP 2: Find sum of probabilities
    for l in range(number_labels):     #loop for sum of probabilities for each label
        for n in range(number_nn):
            sum_of_opinions[l] += ranked_opinions[n][l]
    
    #STEP 3: Rank order the labels
    unique_votes = list(set(sum_of_opinions))   #find unique probabilities
    unique_votes.sort(reverse=True)     #sort probabilities in descending order to get highest-lowest ranking
    loop_counter = 0    #to get correct position for ranking
    ranking: list = [-1]* number_labels
    for uv in unique_votes:
        n_instances = sum_of_opinions.count(uv)     #check for multiple 
        if n_instances==1:
            selected_index = sum_of_opinions.index(uv)  #get the index and assign rank
            ranking[loop_counter] = selected_index
        else:   #multiple labels have same probability
           all_indexes_votes = [i for i in range(len(sum_of_opinions)) if sum_of_opinions[i] == uv] #get indices for same values
           for elem in range(n_instances):
               selected_index = random.choice(all_indexes_votes) #choose random index for ranking
               ranking[elem + loop_counter] = selected_index
               all_indexes_votes.remove(selected_index)  #to avoid replacement during random choice
        
        loop_counter= loop_counter+ n_instances #counter

    return ranking

# random.seed(0)
# agent0 = [0.2,0.2,0.3,0.3]
# agent1 = [0.2,0.2,0.3,0.3]
# agent2 = [0.2,0.2,0.3,0.3]
# print(bordaAggregation([agent0,agent1,agent2]))

# random.seed(0)
# agent0 = [0.6, 0.3, 0.1]
# agent1 = [0.6, 0.1, 0.3]
# agent2 = [0.3,0.6, 0.1]
# print(bordaAggregation([agent0,agent1,agent2]))

random.seed(0)
agent0 = [0.6, 0.3, 0.1]
agent6 = [0.6, 0.3, 0.1]
agent1 = [0.6, 0.1, 0.3]
agent2 = [0.3,0.6, 0.1]
agent5 = [0.3,0.6, 0.1]
agent3 = [0.3,0.6, 0.1]
agent4 = [0.3,0.6, 0.1]

print(bordaAggregation([agent0,agent1,agent2,agent3,agent4,agent5,agent6]))