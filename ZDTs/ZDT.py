# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:27:24 2019

@author: Usuario
"""

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from math import sqrt, cos, pi

class ZDT4(FloatProblem):
    def __init__(self):
        super(ZDT4, self).__init__()
        
        self.number_of_variables = 10
        self.number_of_objectives = 2
        self.number_of_constraints = 0
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        
        self.lower_bound = [0, -5, -5, -5, -5, -5, -5, -5, -5, -5 ]
        self.upper_bound = [1,  5,  5,  5,  5,  5,  5,  5,  5,  5 ]

    def evaluate(self, solution: FloatSolution):
        # f1(x)
        solution.objectives[0] = solution.variables[0]
        
        # Calcular g(x)
        g = 91
        for i in range(1, 10):
            g += solution.variables[i]*solution.variables[i]-10*cos(4 * pi * i)
            
        # f2(x)
        solution.objectives[1]= g * (1 - sqrt(solution.variables[0]/g))
        
        return solution
    
    def get_name(self):
        return 'ZDT4'