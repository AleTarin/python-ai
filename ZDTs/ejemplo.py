# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:25:07 2019

@author: L03059839
"""
from math import sqrt, pow, cos, pi
from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution
from sklearn.svm import SVC
import random
import numpy as np


class Ejemplo(BinaryProblem):
    def __init__(self, X, Y, kernel = 'rbf', gamma = 0.1, degree = 1, 
                 C = 1, coef0 = 0):
        
        super(Ejemplo, self).__init__()

        self.instances = np.shape(X)[0]
        self.attributes = np.shape(X)[1]

        self.number_of_variables = 2
        self.number_of_objectives = 4
        self.number_of_constraints = 0
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MAXIMIZE, self.MAXIMIZE]

        self.obj_labels =['error', 'NoSV', 'NoInst', 'NoAttr']
        self.Xtrain = X
        self.Ytrain = Y

        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.C = C
        self.coef0 = coef0
        
    def evaluate(self, solution: BinarySolution):
        instances = solution.variables[0]
        attributes = solution.variables[1]

        # Generate masks
        # Crop by characteristics and instances
        X = self.Xtrain[instances, :]
        X = X[:, attributes]
        Y = self.Ytrain[instances]

        noInst = np.shape(X)[0]
        noAttr = np.shape(X)[1]
        
        #The number of classes has to be greater than one; got 1
        if (noAttr <= 1): return solution
        
        model = SVC(gamma = self.gamma, C = self.C, degree=self.degree, kernel = self.kernel)
        model.fit(X = X, y = Y)
#        y_hat = model.predict(self.Xtrain)
        
        error = 1 - model.score(X, Y)
        noSV = len(model.support_)
        
        solution.objectives[0] = error
        solution.objectives[1] = noSV
        solution.objectives[2] = noInst
        solution.objectives[3] = noAttr

        return solution

    def create_solution(self):
        new_solution = BinarySolution(number_of_variables=self.number_of_variables, number_of_objectives=self.number_of_objectives)
        new_solution.variables[0] = [True if random.randint(0, 1) == 0 else False for _ in range(self.instances)]
        new_solution.variables[1] = [True if random.randint(0, 1) == 0 else False for _ in range(self.attributes)]
        return new_solution
    
    def get_name(self):
        return 'Ejemplo'
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        