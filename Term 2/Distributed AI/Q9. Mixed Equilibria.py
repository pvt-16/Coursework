from typing import List

def mixedEquilibria(rewards: List[List[tuple]]):
    p = 0
    q = 0

    #rewards = [[(r0,0, r´0,0), (r0,1, r´0,1)], [(r1,0, r´1,0), (r1,1, r´1,1)]]

    # figuring p
    # r00 *p + r01 * (1-p) = r10 * p + r11 * (1-p)
    # rearranging, we get 
    # p = (r11 - r01) / (r00 - r10 - r01 + r11)

    if ((rewards[0][0][0] - rewards[1][0][0] - rewards[0][1][0] + rewards[1][1][0]) == 0 or 
    (rewards[0][0][1] - rewards[0][1][1] - rewards[1][0][1] + rewards[1][1][1]) == 0):
        return ()
    p = (rewards[1][1][0] - rewards[0][1][0]) / (rewards[0][0][0] - rewards[1][0][0] - rewards[0][1][0] + rewards[1][1][0])

    # figuring q
    # r'00 *q + r'10 * (1-q) = r'01 * q + r'11 * (1-q)
    # rearranging, we get 
    # q = (r'11 - r'10) / (r'00 - r'01 - r'10 + r'11)
    q = (rewards[1][1][1] - rewards[1][0][1]) / (rewards[0][0][1] - rewards[0][1][1] - rewards[1][0][1] + rewards[1][1][1])
    return (p,q)

rewards = [[(8,7),(-2,10)],[(5,5),(2,4)]]
print(mixedEquilibria(rewards))

#(0.5, 0.5)
# rewards = [[(5,5),(-1,6)],[(6,-1),(0,0)]]
# print(mixedEquilibria(rewards))