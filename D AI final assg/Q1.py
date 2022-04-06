from itertools import product
from typing import List
# FIXED: THREE potential types of attackers
# ATTACKER is the column player
n_attackers = 3
n_strategies = 2
t = range(n_strategies)
attack_combinations = (sorted(set(product(t,repeat = n_attackers))))
n_attack_combinations = len(attack_combinations)

def nashEquilibria(rewards: List[list]):
    n_cols = len(rewards[0]) 
    n_rows = len(rewards) 
    cell_nash: bool = True
    nash_cells_list :List[tuple] = []
    for r in range(n_rows):
        for c in range(n_cols):
            current_cell = rewards[r][c]
            cell_nash = True
            for r_option in range(n_rows):  #check remaining row options for better reward
                if r_option!= r:
                    potential_reward_cell = rewards[r_option][c]
                    reward_value = potential_reward_cell[0] #for row value
                    if reward_value> current_cell[0]:  #if there is more reward, potential to jump, not nash
                        cell_nash = False
            if cell_nash:    #row value is not nash, we don't need to check column value, since the cell is not nash
                for c_option in range(n_cols):
                    if c_option!= c:
                        potential_reward_cell = rewards[r][c_option]
                        reward_value = potential_reward_cell[1] #for column value
                        if reward_value> current_cell[1]:  #if there is more reward, potential to jump, not nash
                            cell_nash = False
                            break
            
            if cell_nash:   
                nash_cells_list.append((r,c))
    return nash_cells_list

def getSingleNormalFormGame(rewardsB1, rewardsB2, rewardsB3, probs):
    rewards_normform = [[0]* n_attack_combinations for i in range(n_strategies)]
    for r in range(n_strategies):  #rows
        for c in range(n_attack_combinations):     #columns
            selected_attack_strategy = attack_combinations[c]       # format Eg: 000, 010
            (sa1, sa2, sa3) = selected_attack_strategy  #assign selected strategy to each game
            
            # select the cell from each game that has the rewards values for the 
            # selected attackers' strategies (column) based on our strategy(row)
            selected_cell_g1 = rewardsB1[r][sa1]    
            selected_cell_g2 = rewardsB2[r][sa2]
            selected_cell_g3 = rewardsB3[r][sa3]

            #combined for faster calculations in loop
            selected_game_cells = [selected_cell_g1, selected_cell_g2, selected_cell_g3]
            # single normal form game rewards calculations
            comb_val_row = 0; comb_val_col =0
            for n in range(n_attackers):
                comb_val_row += selected_game_cells[n][0] * probs[n]    # row value
                comb_val_col += selected_game_cells[n][1] * probs[n]    # column value
            #set of new rewards
            rewards_normform[r][c] = (comb_val_row, comb_val_col)
    return rewards_normform

def bayesianNashEquilibria(rewardsB1, rewardsB2, rewardsB3, probs):
    rewards_normform = getSingleNormalFormGame(rewardsB1, rewardsB2, rewardsB3, probs)
    nash_cells_list = nashEquilibria(rewards_normform)     #get nash equilibria for new rewards
    nash_equi_list = []
    for cell in nash_cells_list:
        (r,c) = cell
        col_val = ''.join([str(x) for x in list(attack_combinations[c])])
        nash_equi_list.append((str(r), col_val))
    return nash_equi_list

def pureStackelberg(rewardsB1, rewardsB2, rewardsB3, probs):
    rewards_normform = getSingleNormalFormGame(rewardsB1, rewardsB2, rewardsB3, probs)
    follower_strats = []
    for strategy in range(n_strategies):
        #each row is sent as a normal form game to get nash equilibria
        nash_cells_list = nashEquilibria([rewards_normform[strategy]], forPure=1)
        if len(nash_cells_list) != 0:
            lowest_reward_cell = nash_cells_list[0]
            
        # if multiple actions give the same reward for the follower,
        elif len(nash_cells_list) > 1:
            # we will assume the follower will take the one with the lowest reward for the leader
            lowest_reward_cell = (strategy,0)   #assume first cell in row (strategy) has lowest reward
            lowest_leader_reward =rewards_normform[strategy][lowest_reward_cell[1]][0]

            for cell in nash_cells_list:
                cell[0] = strategy  #correction for nashEquilibria method that returns row as 0 
                reward_leader = rewards_normform[cell[0]][cell[1]][0]
                if reward_leader < lowest_leader_reward:
                    lowest_leader_reward = reward_leader
                    lowest_reward_cell = cell
    
        # FORMAT- item 0: ((0,3), (-1.7, -1))
        follower_strats.append([(strategy,lowest_reward_cell[1]), 
        rewards_normform[strategy][lowest_reward_cell[1]]])

    # leader's chance to make a choice based on potential follower reward selections
    #highest_l_reward_position = (0,0)
    #     
    sorted_by_leader_reward = sorted(follower_strats, key=lambda tup: tup[1][0], reverse=True)
    # select the reward with highest value
    # MODIFY TO HAVE MORE THAN ONE
    (r,c) = sorted_by_leader_reward[0][0]
    return [(str(r), ''.join([str(x) for x in list(attack_combinations[c])]) )]

# rewardsB1 = [[(-1,1),(2,-2)],[(-3,4),(1,-1)]]
# rewardsB2 = [[(5,-6),(-4,-5)],[(0,-1),(2,4)]]
# rewardsB3 = [[(3,-3),(0,0)],[(1,2),(-2,1)]]
# probs = [0.5,0.3,0.2]

# print(set(bayesianNashEquilibria(rewardsB1, rewardsB2, rewardsB3, probs))==set([('1', '010')]))


rewardsB1 = [[(-1,1),(2,-2)],[(-3,4),(1,-1)]]
rewardsB2 = [[(5,-6),(-4,-5)],[(0,-1),(2,4)]]
rewardsB3 = [[(3,-3),(0,0)],[(1,2),(-2,1)]]
probs = [0.5,0.3,0.2]

#print(set(bayesianNashEquilibria(rewardsB1, rewardsB2, rewardsB3, probs))==set([('1', '010')]))
#print(set(pureStackelberg(rewardsB1, rewardsB2, rewardsB3, probs))==set([('1', '010')]))

print("other:")
rewards =   [[(0,1),(1,0)],
            [(1,0),(0,1)]]
comb_rewards = rewards            
#print(nashEquilibria([rewards]))
#print(pureStackelberg(rewards))

rewards =   [[(5,5),(-1,6)],
            [(6,-1),(0,0)]]
comb_rewards = rewards            
print(nashEquilibria(rewards))
#print(pureStackelberg(rewards))

rewards =   [[(50,50),(-1,6)],
            [(6,-1),(0,0)]]
comb_rewards = rewards
#print(bayesianNashEquilibria(rewards))
#print(pureStackelberg(rewards))