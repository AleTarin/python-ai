################
#  Alejandro De la Cruz Tarin
#  A00816503
# 
#  “Apegandome al Codigo de Etica de los Estudiantes del Tecnologico de Monterrey, 
#  me comprometo a que mi actuaci´on en este examen este regida por la honestidad
#  academica.”
####################

# Imports
from scipy._lib._util import check_random_state
import numpy as np
from numpy import genfromtxt


class EvolutionaryConstrainedSVM(object):
    
    def __init__(self, Xtrain, Ytrain, kernel = 'linear', gamma = 0.1, degree = 1, 
                 C = 1, coef0 = 0, generations = 1000, populationSize = 100, 
                 constraint = 'feasibilityRules', CR = 0.9, F = 0.1, seed = None, 
                 threshold = 0.0001, feasibilityProbability = 0.5):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.seed = check_random_state(seed)
        self.generations = generations
        self.popsize = populationSize
        self.constraint = constraint
        self.CR = CR
        self.F = F
        self.threshold = threshold
        self.fp = feasibilityProbability
        self.C = C
        self.coef0 = coef0
        self.mutationRate = F
        self.crossoverRate = CR
        self.solution = None
        
    def _coevolution(self, parents, fparents,offspring, 
                             foffspring, ):
        # segundo if esta mal indentado
        # phi son las restricciones
        
        return 0
    
        
    def __compute_kernel(self, X, Y):
        if self.kernel == 'linear':
            return self._linear_kernel(X, Y)
        elif self.kernel == 'polynomial':
            return self._polynomial_kernel(X, Y)
        elif self.kernel == 'rbf':
            return self._rbf_kernel(X, Y)
        else:
            raise ValueError("Please select a valid kernel function")
    
    def __constraint_handler(self, parents, fparents, cparents, offspring, 
                             foffspring, coffspring):
        return self._coevolution(parents, fparents, offspring, 
                     foffspring)

    def __fitness(self, alpha):        
        # alfa = 1 x m
        # y = m x 1
        # k = m x m
        
        alphaY = np.multiply(alpha, np.transpose(self.Ytrain))
        aux = np.dot(alphaY,self.k)
        aux2 = np.dot(aux, np.transpose(alphaY))
        aux3 = np.dot(alpha, np.ones(np.shape(Ytrain)[0]))
        return  (0.5 * aux2 * - aux3)
    
    def __constraint(self, alpha):
        return np.dot(alpha, self.Ytrain)

    def train(self):
        bounds = np.array(np.multiply([0, self.C], np.ones((np.shape(self.Xtrain)[0],2))))
        numVars = bounds.shape[0]

        # Calcular el kernel del entrenamiento 
        self.k = self.__compute_kernel(self.Xtrain, self.Xtrain)

        if not np.all((bounds[:,1] - bounds[:,0]) > 0):
            raise ValueError("Error: lower bound greater than upper bound")
        
        # Generar poblacion random
        parents = np.random.uniform(0, self.C, size=(self.popsize, numVars))  
        
        # Calcular fitness para cada individio
        fparents = np.asarray([self.__fitness(ind) for ind in parents])
        cparents = np.asarray([self.__constraint(ind) for ind in parents])
        
        currentGeneration = 0
        # While a stop criterion is not met do ...
        while currentGeneration < self.generations + 1:
            print ('GENERATION:', currentGeneration)
            # Para cada elemento de la poblacion
            for j in range(self.popsize):
                # Seleccionar individuos de P (t)
                idxs = [idx for idx in range(self.popsize) if idx != j]
                
                # Dividir el vector en tres random ???
                a, b, c = parents[np.random.choice(idxs, 3, replace = False)]

                # Para cada individuo, aplicar directional mutation
                mutant = np.clip(a + self.mutationRate * (b - c), 0, 1)
                
                # Generar los puntos de cross over, siempre que sean menores al ratio y aplica un crossover random uniforme
                cross_points = np.random.rand(numVars) < self.crossoverRate
                
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, numVars)] = True
                    
                # Apply crossover to produce a child
                offspring = np.where(cross_points, mutant, parents[j])
                
                # Obtener el fitness y el contraint del hijo
                foffspring = self.__fitness(offspring)
                coffspring = self.__constraint(offspring)

                # if the child individual is better than the current one then
                if self.__constraint_handler(parents[j], fparents[j], cparents[j], offspring, foffspring, coffspring) == 1:
                    # Hacer el intercambio
                    fparents[j] = foffspring
                    parents[j] = offspring
                                        
            currentGeneration += 1

        
        idxBest = np.argmin(fparents)
        
        
        print('Best:', parents[np.argmin(idxBest)])
        print('Min:', np.min(fparents))
        self.solution = parents[np.argmin(idxBest)]
        
        print(parents[np.argmin(idxBest)].shape)
        
        return parents[np.argmin(idxBest)]
    

    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        return None
    
# Mandar a llamar al AE
sol = EvolutionaryConstrainedSVM()
sol.train()


