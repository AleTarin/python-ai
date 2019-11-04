# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:25:07 2019

@author: L03059839
"""
from math import sqrt, pow, cos, pi
from jmetal.core.problem import FloatProblem, BinaryProblem, IntegerProblem
from jmetal.core.solution import FloatSolution, IntegerSolution, BinarySolution
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy._lib._util import check_random_state
from random import random, choices, randint, uniform
import numpy as np


class SVM_Problem(IntegerProblem):
    
    def __init__(self, X, Y):
        
        super(SVM_Problem, self).__init__()
        
        self.number_of_variables = 5
        self.number_of_objectives = 4
        self.number_of_constraints = 0

        self.instances = np.shape(X)[0]
        self.attributes = np.shape(X)[1]

        self.Xtrain = X
        self.Ytrain = Y

        #                   gamma=0 , C=1    , coef0=2, degree, k 
        self.lower_bound = [2**(-10), 1      , 2**(-3), 0     , 0]
        self.upper_bound = [2**(3  ), 2**(10), 2**(10), 10    , 3]

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels     = ['error'      , 'NoSV'       , 'Instances'  , 'Attributes' ]

        self.model = None
         
    def evaluate(self, solution: IntegerProblem):

        # print (*solution.variables, sep=", ")
      
        solution.masks  = \
            [[True if randint(0, 1) == 0 else False for _ in range(self.instances)],
            [True if randint(0, 1) == 0 else False for _ in range(self.attributes)]]

        # Get our variables
        gamma        = solution.variables[0]
        C            = solution.variables[1]
        coef0        = solution.variables[2]
        degree       = solution.variables[3]
        kernel       = solution.variables[4] 

        instances    = solution.masks[0]
        attributes   = solution.masks[1]
        
        # Generate masks
        # Crop by characteristics and instances
        X = self.Xtrain[instances, :]
        X = X[:, attributes]
        Y = self.Ytrain[instances]

        noInst = np.shape(X)[0]
        noAttr = np.shape(X)[1]
        
        # Avoid solutions that don't have attributes or instances
        if (noAttr == 0 or noInst == 0): return solution
        
        # Train our SVM
        self.model = SVM(Xtrain=X, Ytrain=Y, gamma = gamma, C = C, kernel = kernel, coef0=coef0, degree=degree).train()

        error = 1 - self.model.score(X, Y)
        noSV = len(self.model.support_)
        
        solution.objectives[0] = error
        solution.objectives[1] = noSV
        solution.objectives[2] = noInst
        solution.objectives[3] = noAttr
        
        return solution

    def get_name(self):
        return 'SVM_Problem'
        
        
class SVM(object):
    def __init__(self, Xtrain, Ytrain, kernel=3, gamma=0.1, degree=1,
                 C=1, coef0=0, maxEvaluations=100000, populationSize=100,
                 seed=None, ):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.kernel = kernel
        # ValueError: The gamma value of 0.0 is invalid. Use 'auto' to set gamma to a value of 1 / n_features.        
        self.gamma = gamma if gamma != 0 else 'auto'
        self.degree = degree
        self.seed = check_random_state(seed)
        self.maxEvaluations = maxEvaluations
        self.popsize = populationSize
        self.C = C
        self.coef0 = coef0
        self.model = None
        self.y_pred = None

        # Not including poly kernel, that made the times much larger
        self.kernelTypes = ['linear', 'rbf', 'sigmoid', 'precomputed']


    def train(self):
        # Create a svm Classifier
        k = self.kernelTypes[ int(self.kernel) ]
        self.model = SVC(kernel=k, C=self.C, degree=self.degree, coef0=self.coef0,
                                gamma=self.gamma, random_state=self.seed, verbose=False)

        # Train and return the model using the training sets
        self.model.fit(self.Xtrain, self.Ytrain)
        return self.model

    def predict(self, Xtest):
        # Predict the response for test dataset
        self.y_pred = self.model.predict(Xtest)
        return self.y_pred

    def accuracy(self, Ytest):
        # Model Accuracy: how often is the classifier correct?
        print("Accuracy:", accuracy_score(Ytest, self.y_pred))
        # print(confusion_matrix(Ytest, self.y_pred))
        # print(classification_report(Ytest, self.y_pred))

    def get_name(self):
        return 'SVM'
        
        
        
        
        
        
        
        
        