from scipy._lib._util import check_random_state

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import sys
import os
import matplotlib.pyplot as plt

import numpy as np
from numpy import genfromtxt

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.moead import MOEAD
from jmetal.lab.visualization import Plot, InteractivePlot
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.problem import ZDT1, ZDT3
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.solutions import read_solutions, print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations
from ZDTs.SVM import SVM_Problem
from ZDTs.SVM import SVM
from sklearn.preprocessing import normalize



class MultiObjectiveTest(object):

    def __init__(self, Xtrain, Ytrain,  kernel='rbf', gamma=0.1, degree=1,
                 C=1, coef0=0, maxEvaluations=100000, populationSize=100, seed=None ):
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

        self.model = None

    def train(self):
        problem = SVM_Problem(X=self.Xtrain, Y=self.Ytrain)
        #problem.reference_front = read_solutions(filename='resources/reference_front/ZDT1.pf')

        max_evaluations = self.maxEvaluations
        algorithm = NSGAII(
            problem=problem,
            population_size=self.popsize,
            offspring_population_size=self.popsize,
            mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
            crossover=SBXCrossover(probability=1.0, distribution_index=20),
            termination_criterion=StoppingByEvaluations(max=max_evaluations)
        )

        algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
        #algorithm.observable.register(observer=VisualizerObserver(reference_front=problem.reference_front))

        algorithm.run()
        front = algorithm.get_result()

        # Plot front
        plot_front = Plot(plot_title='Pareto front approximation',
                            reference_front=None, axis_labels=problem.obj_labels)
        plot_front.plot(front, label=algorithm.label, filename=algorithm.get_name())

        # Plot interactive front
        plot_front = InteractivePlot(plot_title='Pareto front approximation', axis_labels=problem.obj_labels)
        plot_front.plot(front, label=algorithm.label,filename=algorithm.get_name())

        # Save results to file
        print_function_values_to_file(front, 'FUN.' + algorithm.label)
        print_variables_to_file(front, 'VAR.' + algorithm.label)
        print('Algorithm (continuous problem): ' + algorithm.get_name())
        
        print("-----------------------------------------------------------------------------")
        print('Problem: ' + problem.get_name())
        print('Computing time: ' + str(algorithm.total_computing_time))
        
        # Get normalized matrix of results
        normed_matrix = normalize(list(map(lambda result : result.objectives, front )))
        
        # Get the sum of each objective results and select the best (min) 
        scores = list(map(lambda item : sum(item), normed_matrix ))
        solution = front[ scores.index( min(scores))]
        
        # Get our variables
        self.gamma        = solution.variables[0]
        self.C            = solution.variables[1]
        self.coef0        = solution.variables[2]
        self.degree       = solution.variables[3]
        self.kernel       = solution.variables[4]

        self.instances    = solution.masks[0]
        self.attributes   = solution.masks[1]
        
        # Select pick a random array with length of the variable
        X = self.Xtrain[self.instances, :]       
        X = X[:, self.attributes]
        Y = self.Ytrain[self.instances]
        
        # Contruct model
        self.model = SVM( Xtrain=X, Ytrain=Y,  kernel=self.kernel, C=self.C, degree=self.degree, coef0=self.coef0,
                            gamma=self.gamma, seed=self.seed ).train()

        print ('Objectives: ', *solution.objectives, sep=", ")
        # write your code here
        return self.model

    def predict(self, Xtest):
        # Predict the response for test dataset
        X = Xtest[:, self.attributes]
        self.y_pred = self.model.predict(X)
        return self.y_pred

    def accuracy(self, Ytest):
        # Model Accuracy: how often is the classifier correct
        print("Accuracy:", accuracy_score(Ytest, self.y_pred))


# Get data fron csv's
Xtrain = genfromtxt(
    fname='./datasetCI2019/echocardiogram/Xtrain.csv', delimiter=',', dtype=None)
Ytrain = genfromtxt(
    fname='./datasetCI2019/echocardiogram/Ytrain.csv', delimiter=',', dtype=None)
Xtest = genfromtxt(
    fname='./datasetCI2019/echocardiogram/Xtest.csv', delimiter=',', dtype=None)
Ytest = genfromtxt(
    fname='./datasetCI2019/echocardiogram/Ytest.csv', delimiter=',', dtype=None)


# Test1: echocardiogram :
# My   Acc: 0.7857142857142857
# Prof Acc: 0.8

# Test2: fertility :
# My   Acc: 0.9
# Prof Acc: 0.9

test = MultiObjectiveTest(Xtrain=Xtrain, Ytrain=Ytrain, maxEvaluations=1000)
model = test.train()
test.predict(Xtest)
test.accuracy(Ytest)
