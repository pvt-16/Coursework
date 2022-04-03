from typing import List

def nashEquilibria(rewards: List[list]):
    n_cols = len(rewards[0])    #For 2 agents, rows and columns are 2
    n_rows = n_cols
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
            
            if cell_nash:   #if cell is nash equilibria, we add the index to the list
                nash_cells_list.append((r,c))
    return nash_cells_list


rewards = [[(5,5),(-1,6)],[(6,-1),(0,0)]]
pure_strategies_nash_eq = nashEquilibria(rewards)
print(set(pure_strategies_nash_eq)==set([(1,1)]))
#True