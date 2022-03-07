# #coursework462.py
# def howManyIterations(x, epsilon): #return number of iterations taken to reach value less than epsilon
#     count =0
#     errorVal =x
#     if errorVal < epsilon or x>= 0.5 : #Schapire's graph -we only consider x<0.5
#         return False

#     while (errorVal > epsilon):
#         count = count+1 #counter 
#         x = errorVal #the calculated value is the new x valye
#         errorVal = (3*x*x - (2*x*x*x))  #Equation for error as shown by Schapire
#     return count #Total number of iterations
    
# x = 0.4
# epsilon = 0.2
# print(howManyIterations(x,epsilon))

