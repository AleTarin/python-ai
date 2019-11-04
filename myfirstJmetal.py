# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:33:43 2019

@author: Alejandro De la Cruz
"""

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.lab.visualization import Plot, InteractivePlot
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.problem import ZDT1, ZDT3, ZDT4, ZDT6
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.solutions import read_solutions, print_function_values_to_screen
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = ZDT3()
    problem.reference_from = read_solutions(filename = 'mysolucionbonita.pf')
    max_evaluations = 25000
    algorithm = NSGAII(
        problem = problem,
        population_size = 100,
        offspring_population_size = 100,
        mutation = PolynomialMutation( probability=1.0/problem.number_of_variables, distribution_index =20),
        crossover = SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion = StoppingByEvaluations(max = max_evaluations)
    )    
    algorithm.observable.register(observer = ProgressBarObserver(max= max_evaluations))
    algorithm.observable.register(observer = VisualizerObserver(reference_front = problem.reference_front))
    
    algorithm.run()
    front = algorithm.get_result()
      # pareto_front = FrontPlot(plot_title='NSGAII', axis_labels=problem.obj_labels)
    #pareto_front.plot(front, reference_front=problem.reference_front)
    #areto_front.to_html(filename='NSGAII')

    #print_function_values_to_file(front, 'FUN.NSGAII')
    #print_variables_to_file(front, 'VAR.NSGAII')

    print_function_values_to_file(front, 'FUN.NSGAII')
    print_variables_to_file(front, 'VAR.NSGAII')
    