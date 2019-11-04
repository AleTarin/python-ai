from scipy._lib._util import check_random_state
from ZDTs.ejemplo import Ejemplo

import numpy as np
from numpy import genfromtxt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.moead import MOEAD
from jmetal.lab.visualization import Plot, InteractivePlot
from jmetal.operator import SBXCrossover, PolynomialMutation, SPXCrossover, BitFlipMutation
from jmetal.problem import ZDT1, ZDT3
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.solutions import read_solutions, print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations
from sklearn.preprocessing import normalize

class MultiObjectiveTest(object):
    
    def __init__(self, Xtrain, Ytrain, kernel = 'rbf', gamma = 0.1, degree = 1, 
                 C = 1, coef0 = 0, maxEvaluations = 10000, populationSize = 100, 
                 seed = None,):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.C = C
        self.coef0 = coef0
        self.seed = check_random_state(seed)
        self.maxEvaluations = maxEvaluations
        self.popsize = populationSize

        self.model = None
    
    def train(self):
        problem = Ejemplo(X=self.Xtrain, Y=self.Ytrain, kernel=self.kernel, gamma=self.gamma, degree=self.degree, C=self.C, coef0=self.coef0 )
        #problem.reference_front = read_solutions(filename='resources/reference_front/ZDT1.pf')

        max_evaluations = self.maxEvaluations
        algorithm = NSGAII(
            problem=problem,
            population_size=self.popsize,
            offspring_population_size=self.popsize,
            mutation=BitFlipMutation(probability=1.0 / np.shape(self.Xtrain)[0]),
            crossover=SPXCrossover(probability=1.0),
            termination_criterion=StoppingByEvaluations(max=max_evaluations)
        )

        algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
        #algorithm.observable.register(observer=VisualizerObserver(reference_front=problem.reference_front))

        algorithm.run()
        front = algorithm.get_result()

        # Plot front
        plot_front = Plot(plot_title='Pareto front approximation', reference_front=None, axis_labels=problem.obj_labels)
        plot_front.plot(front, label=algorithm.label, filename=algorithm.get_name())

        # Plot interactive front
        plot_front = InteractivePlot(plot_title='Pareto front approximation', axis_labels=problem.obj_labels)
        plot_front.plot(front, label=algorithm.label,filename=algorithm.get_name())

        # Save results to file
        print_function_values_to_file(front, 'FUN.' + algorithm.label)
        print_variables_to_file(front, 'VAR.' + algorithm.label)
        print('Algorithm (continuous problem): ' + algorithm.get_name())

        # Get normalized matrix of results
        normed_matrix = normalize(list(map(lambda result : result.objectives, front )))
        
        # Get the sum of each objective results and select the best (min) 
        scores = list(map(lambda item : sum(item), normed_matrix ))
        solution = front[ scores.index( min(scores))]
        
        self.instances =  solution.variables[0]
        self.attributes = solution.variables[1]

        # Generate masks
        # Crop by characteristics and instances
        X = self.Xtrain[self.instances, :]
        X = X[:, self.attributes]
        Y = self.Ytrain[self.instances]

        self.model = SVC(gamma = self.gamma, C = self.C, degree=self.degree, kernel = self.kernel)
        self.model.fit(X = X, y = Y)
        
        # write your code here
        return self.model
        
    def predict(self, Xtest):
        # Write your code here
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
  
test = MultiObjectiveTest(Xtrain=Xtrain, Ytrain=Ytrain)
test.train()
test.predict(Xtest)
test.accuracy(Ytest)
