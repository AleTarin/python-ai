from scipy._lib._util import check_random_state

class MultiObjectiveTest(object):
    
    def __init__(self, Xtrain, Ytrain, kernel = 'rbf', gamma = 0.1, degree = 1, 
                 C = 1, coef0 = 0, maxEvaluations = 100000, populationSize = 100, 
                 seed = None,):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.seed = check_random_state(seed)
        self.maxEvaluations = maxEvaluations
        self.popsize = populationSize
        self.C = C
        self.coef0 = coef0
    
    
    def train(self):
        # write your code here
        return 0
        
    def predict(self, Xtest):
        # Write your code here
        return 0 
        
    