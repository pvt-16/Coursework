#coursework462.py

#Q2.
def howManyIterations(x, epsilon): #return number of iterations taken to reach error less than epsilon
    count = 0
    errorVal = x
    x_org = x
    if x <= epsilon:  #x value is already less than epsilon
        return 0
    # value of x is bounded 3x^2 - 2x^3 which converges to 0 at x=1.5 and epsilon becomes <=0 after that
    # According to Schapire's paper, 0 < epsilon. so we return false for epsilon <=0 
    # x=1 gives error value as 1. The error doesn't change after that. Hence we add that check
    # as per the equation graph, Between x = 0.5 and x = 1, error just increases and doesn't reach below epsilon
    if x<0  or epsilon <= 0:
        return False
    while (errorVal > epsilon):
        count = count+1 #counter 
        x = errorVal # the calculated value is the new x value
        errorVal = (3*x*x - (2*x*x*x))  #Equation for error as shown by Schapire
        if errorVal > x_org and x_org<1:   #if error is increasing, it'll never come done to epsilon since epsilon < x
            return False
        # This can be seen between x = 0.5 and x = 1, error just increases and doesn't reach below epsilon
    return count #Total number of iterations
        
x = 0.5
epsilon = 0.01
print(howManyIterations(x,epsilon)) 