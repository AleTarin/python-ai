import numpy as np
from scipy._lib._util import check_random_state, MapWrapper  

def genericEA(func, bounds, args = None, crossover = 'arithmetic', mutation = 'uniform', 
              generations = 1000, populationSize = 100, recombinationRate = 0.9, 
              mutationRate = 0.1, seed = None):
    
    with EvolutionaryAlgorithmSolver(func, bounds, args, crossover, mutation, 
                                     generations, populationSize, 
                                     recombinationRate, mutationRate, 
                                     seed) as solver:
        solution = solver.solve()
    
    return solution
    

class EvolutionaryAlgorithmSolver(object):
    
    def __init__(self, func, bounds, args = None, crossover = 'arithmetic', 
                 mutation = 'uniform', generations = 1000, populationSize = 100, 
                 recombinationRate = 0.9, mutationRate = 0.1, seed = None):
        self.func = _FunctionWrapper(func, args)
        self.bounds = np.array(bounds)
        self.seed = check_random_state(seed)
        self.crossoverOp = crossover
        self.mutationOp = mutation
        self.generations = generations
        self.popsize = populationSize
        self.recombinationrate = recombinationRate
        self.mutationrate = mutationRate
        
        self._mapwrapper = MapWrapper(4)
        
    def tournament(self, parents, fitness):
        rng = self.seed
        idx = np.argsort(rng.uniform(size = (self.popsize, self.popsize)))[:,0:4]
        idx_1 = fitness[idx[:,0]] < fitness[idx[:,1]]
        idx_2 = fitness[idx[:,2]] < fitness[idx[:,3]]
        idx_parent_1 = idx[:,1]
        idx_parent_1[idx_1] = idx[idx_1,0]
        idx_parent_2 = idx[:,3]
        idx_parent_2[idx_2] = idx[idx_2,2]
        return (parents[idx_parent_1,:], parents[idx_parent_2,:])
    
    def arithmetic(self, parents):
        rng = self.seed
        size = np.shape(parents[0])
        alpha = rng.uniform(size = size)
        mask = rng.choice([False, True], size = (size[0]), 
                          p = [self.recombinationrate, 1-self.recombinationrate])
        offspring = np.multiply(alpha, parents[0]) + np.multiply(1-alpha, 
                               parents[1])
        offspring[offspring > 1] = 1
        offspring[offspring < 0] = 0
        offspring[mask] = parents[0][mask]
        return offspring
    
    def uniform(self, offspring):
        rng = self.seed
        size=np.shape(offspring)
        mutation = rng.uniform(size = size)
        mask = rng.choice([True, False], size = size, p = [self.mutationrate, \
                          1-self.mutationrate])
        offspring[mask] = mutation[mask]
        return offspring
    
    def __crossover(self, parents):
        if self.crossoverOp == 'arithmetic':
            return self.arithmetic(parents)
        else:
            raise ValueError("Please select a valid crossover strategy")
    
    def __mutation(self, offspring):
        if self.mutationOp == 'uniform':
            return self.uniform(offspring)
        else:
            raise ValueError("Please select a valid mutation strategy")
    
    def __reproduce(self, parents):
        return self.__mutation(self.__crossover(parents))
    
    def __survivalSelection(self, parents, fparents, offspring, foffspring):
        mergedPop = np.vstack((parents, offspring))
        mergedFit = np.hstack((fparents, foffspring))
        
        idx = np.argsort(mergedFit)[0:self.popsize]
        
        return mergedPop[idx,:], mergedFit[idx]
    
    def __scaleParameters(self, population):
        scaled = np.multiply((self.bounds[:,1] - self.bounds[:,0]), population)
        return scaled + self.bounds[:,0]
        
    def solve(self):
        rng = self.seed
        
        numVars = np.shape(self.bounds)[0]
        
        # Check the bounds
        if not np.all((self.bounds[:,1] - self.bounds[:,0]) > 0):
            raise ValueError("Error: lower bound greater than upper bound")
        
        pop = rng.uniform(size=(self.popsize,numVars))
        
        fitness = self.func(self.__scaleParameters(pop))
        
        currentGeneration = 0
        while currentGeneration < self.generations + 1:
            parents = self.tournament(pop, fitness)
            offspring = self.__reproduce(parents)
            foffspring = self.func(self.__scaleParameters(offspring))
            
            pop, fitness = self.__survivalSelection(pop, fitness, offspring,
                                                    foffspring)
            print('The best value in generation '+str(currentGeneration)+ ' is '+str(np.min(fitness)))
            currentGeneration += 1
        
        idxBest = np.argmin(fitness)
        return (self.__scaleParameters(pop[idxBest,:]), fitness[idxBest])
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self._mapwrapper.close()

class _FunctionWrapper(object):
    
    def __init__(self, f, args):
        self.f = f
        self.args = [] if args is None else args

    def __call__(self, x):
        return self.f(x, *self.args)

# A toy optimization problem    
def rastringin(x):
    n = np.shape(x)[1]
    return (10 * n) + np.sum((x**2) - 10 * np.cos(2 * np.pi * x), axis = 1)


# Does it work? 
bounds = np.multiply([-5.12, 5.12], np.ones((1000,2)))
solution = genericEA(rastringin, bounds, generations = 1000, seed = 123, recombinationRate=1, populationSize=7000)




Ir al contenido
Cómo usar Gmail con lectores de pantalla

tarea
Recibidos
x

Fabiola Palacios
Adjuntosvie., 6 sep. 12:48 (hace 5 días)
 

ale tarin <ale.tarin10@gmail.com>
Adjuntos
lun., 9 sep. 08:43 (hace 2 días)
para laylatame



---------- Forwarded message ---------
De: Fabiola Palacios <sfabiola.palacios@gmail.com>
Date: vie., 6 de sep. de 2019 a la(s) 12:48
Subject: tarea
To: <ale.tarin10@gmail.com>




Área de archivos adjuntos

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:07:35 2019

@author: FabiolaPalacios
"""

import numpy as np
from scipy._lib._util import check_random_state, MapWrapper  

def genericEA(func, bounds, args = None, crossover = 'arithmetic', mutation = 'uniform', 
              generations = 1000, populationSize = 100, recombinationRate = 0.9, 
              mutationRate = 0.1, seed = None):
    
    with EvolutionaryAlgorithmSolver(func, bounds, args, crossover, mutation, 
                                     generations, populationSize, 
                                     recombinationRate, mutationRate, 
                                     seed) as solver:
        solution = solver.solve()
    
    return solution
    

class EvolutionaryAlgorithmSolver(object):
    
    def __init__(self, func, bounds, args = None, crossover = 'arithmetic', 
                 mutation = 'uniform', generations = 1000, populationSize = 100, 
                 recombinationRate = 0.9, mutationRate = 0.1, seed = None):
        self.func = _FunctionWrapper(func, args)
        self.bounds = np.array(bounds)
        self.seed = check_random_state(seed)
        self.crossoverOp = crossover
        self.mutationOp = mutation
        self.generations = generations
        self.popsize = populationSize
        self.recombinationrate = recombinationRate
        self.mutationrate = mutationRate
        
        self._mapwrapper = MapWrapper(4)
        
    def tournament(self, parents, fitness):
        rng = self.seed
        idx = np.argsort(rng.uniform(size = (self.popsize, self.popsize)))[:,0:4]
        idx_1 = fitness[idx[:,0]] < fitness[idx[:,1]]
        idx_2 = fitness[idx[:,2]] < fitness[idx[:,3]]
        idx_parent_1 = idx[:,1]
        idx_parent_1[idx_1] = idx[idx_1,0]
        idx_parent_2 = idx[:,3]
        idx_parent_2[idx_2] = idx[idx_2,2]
        return (parents[idx_parent_1,:], parents[idx_parent_2,:])
    
    def arithmetic(self, parents):
        rng = self.seed
        size = np.shape(parents[0])
        alpha = rng.uniform(size = size)
        mask = rng.choice([False, True], size = (size[0]), 
                          p = [self.recombinationrate, 1-self.recombinationrate])
        offspring = np.multiply(alpha, parents[0]) + np.multiply(1-alpha, 
                               parents[1])
        offspring[offspring > 1] = 1
        offspring[offspring < 0] = 0
        offspring[mask] = parents[0][mask]
        return offspring
    
      #BLX
    def blx(self, parents):
        rng = self.seed
        size = np.shape(parents[0])
        alpha = rng.uniform(size = size)
        mask = rng.choice([False, True], size = (size[0]), 
                          p = [self.recombinationrate, 1-self.recombinationrate])
        d = np.absolute(parents[0]-parents[1])
        alphaD = np.multiply(alpha, d) 
        q = np.minimum(parents[0],parents[1])-alphaD
        r=np.maximum(parents[0],parents[1])+alphaD
        
        offspring= rng.uniform(low=q,high=r)  
        
        offspring[offspring > 1] = 1
        offspring[offspring < 0] = 0
        offspring[mask] = parents[0][mask]
        return offspring
    
        #SBX
    def sbx(self,parents):
        rng = self.seed
        size = np.shape(parents[0])
      
        mask = rng.choice([False, True], size = (size[0]), 
                          p = [self.recombinationrate, 1-self.recombinationrate])
        
        u=rng.uniform(size=size)
        n=10
        
         
        b=np.power(2*u,1/n)
            #(2*u)**(1/(n+1))
       
        b[u>.5]=np.power(2*(1-u[u>.5]),1/n)
        offspring=.5*(parents[0]+parents[1])-.5*np.multiply(b,parents[0]-parents[1])
        
            
               
        offspring[offspring > 1] = 1
        offspring[offspring < 0] = 0
        offspring[mask] = parents[0][mask]
        return offspring
    
    def uniform(self, offspring):
        rng = self.seed
        size=np.shape(offspring)
        mutation = rng.uniform(size = size)
        mask = rng.choice([True, False], size = size, p = [self.mutationrate, \
                          1-self.mutationrate])
        offspring[mask] = mutation[mask]
        return offspring
    
    def boundary(self,offspring):
        rng = self.seed
        size=np.shape(offspring)
        mutation =np.round(rng.uniform(size = size))
        
        mask = rng.choice([True, False], size = size, p = [self.mutationrate, \
                          1-self.mutationrate])
        offspring[mask] = mutation[mask]
        return offspring
    
    
    def __crossover(self, parents):
        if self.crossoverOp == 'blx':
            return self.blx(parents)
        elif self.crossoverOp == 'sbx':
            return self.sbx(parents)
        elif self.crossoverOp == 'arithmetic':
            return self.arithmetic(parents)
        else:
            raise ValueError("Please select a valid crossover strategy")
    
    def __mutation(self, offspring):
        if self.mutationOp == 'uniform':
            return self.uniform(offspring)
        elif self.mutationOp == 'boundary':
            return self.boundary(offspring)
        else:
            raise ValueError("Please select a valid mutation strategy")
            
    
    def __reproduce(self, parents):
        return self.__mutation(self.__crossover(parents))
    
    def __survivalSelection(self, parents, fparents, offspring, foffspring):
        mergedPop = np.vstack((parents, offspring))
        mergedFit = np.hstack((fparents, foffspring))
        
        idx = np.argsort(mergedFit)[0:self.popsize]
        
        return mergedPop[idx,:], mergedFit[idx]
    
    def __scaleParameters(self, population):
        scaled = np.multiply((self.bounds[:,1] - self.bounds[:,0]), population)
        return scaled + self.bounds[:,0]
        
    def solve(self):
        rng = self.seed
        
        numVars = np.shape(self.bounds)[0]
        
        # Check the bounds
        if not np.all((self.bounds[:,1] - self.bounds[:,0]) > 0):
            raise ValueError("Error: lower bound greater than upper bound")
        
        pop = rng.uniform(size=(self.popsize,numVars))
        
        fitness = self.func(self.__scaleParameters(pop))
        
        currentGeneration = 0
        while currentGeneration < self.generations + 1:
            parents = self.tournament(pop, fitness)
            offspring = self.__reproduce(parents)
            foffspring = self.func(self.__scaleParameters(offspring))
            
            pop, fitness = self.__survivalSelection(pop, fitness, offspring,
                                                    foffspring)
            print('The best value in generation '+str(currentGeneration)+ ' is '+str(np.min(fitness)))
            currentGeneration += 1
        
        idxBest = np.argmin(fitness)
        return (self.__scaleParameters(pop[idxBest,:]), fitness[idxBest])
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self._mapwrapper.close()

class _FunctionWrapper(object):
    
    def __init__(self, f, args):
        self.f = f
        self.args = [] if args is None else args

    def __call__(self, x):
        return self.f(x, *self.args)

# A toy optimization problem    
def rastringin(x):
    n = np.shape(x)[1]
    return (10 * n) + np.sum((x**2) - 10 * np.cos(2 * np.pi * x), axis = 1)


# Does it work? 
bounds = np.multiply([-5.12, 5.12], np.ones((10,2)))
solution = genericEA(rastringin, bounds, generations = 100,crossover='blx', seed = 1, recombinationRate=1, populationSize=1000)
