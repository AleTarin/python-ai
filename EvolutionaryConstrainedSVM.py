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
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import linear_kernel
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
        
    
    # Kernels calculados con sklearn ---------------------------
    def _linear_kernel(self, X, Y):
        # np.dot(X, np.transpose(Y))
        return linear_kernel(X,Y)
    
    def _polynomial_kernel(self, X, Y):
        # np.power(np.dot(X, np.transpose(Y) + self.coef0),self.degree)
        return polynomial_kernel(X,Y,self.degree, self.coef0)
    
    
    def _rbf_kernel(self, X, Y):
        #    np.power(math.e, -self.gamma*np.linalg.norm(X - Y, 2))
        return rbf_kernel(X,Y, self.gamma)
    
    ###---------------------------------------------------
    
    def _feasibility_rules(self, parents, fparents, cparents, offspring,foffspring, coffspring):
        if cparents == 0 and coffspring != 0:
            return 0
        if coffspring == 0 and cparents != 0:
            return 1
        # Si ambos son 0 entonces ambos son factibles ir por el mejor fitness
        if cparents == 0 and coffspring == 0:
            # mejor parent por fitness
            if fparents > foffspring:
                return 0
            else:
                # mejor offspring por fitness
                return 1
        # Si los constraints son infactibles.
        if cparents != 0 and coffspring != 0:
            # mejor parent por fitness
            if cparents < coffspring:
                return 0
            else:
                # mejor offspring por fitness
                return 1
    
    def _epsilon_constraint(self, parents, fparents, cparents, offspring, 
                             foffspring, coffspring):
        # Write your code here
        return 0
    
    def _stochastic_ranking(self, parents, fparents, cparents, offspring, 
                             foffspring, coffspring):
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
        if self.constraint == 'feasibilityRules':
            # fitness del parent, fitness del hijo
            return self._feasibility_rules(parents, fparents, cparents, offspring, 
                             foffspring, coffspring)
        elif self.constraint == 'epsilonConstraint': #facil 
            return self._epsilon_constraint(parents, fparents, cparents, offspring, 
                             foffspring, coffspring)
        elif self.constraint == 'stochasticRanking':
            return self._stochastic_ranking(parents, fparents, cparents, offspring, 
                             foffspring, coffspring)
        else:
            raise ValueError("Please select a valid constraint-handling technique")
    

    # BLX
    def __crossover(self, parents):
        recombinationrate = 0.9
        rng = self.seed
        size = np.shape(parents[0])
        alpha = rng.uniform(size = size)
        mask = rng.choice([False, True], size = (size[0]), 
                          p = [recombinationrate, 1-recombinationrate])
        d = np.absolute(parents[0]-parents[1])
        alphaD = np.multiply(alpha, d) 
        q = np.minimum(parents[0],parents[1])-alphaD
        r=np.maximum(parents[0],parents[1])+alphaD
        
        offspring= rng.uniform(low=q,high=r)  
        
        offspring[offspring > 1] = 1
        offspring[offspring < 0] = 0
        offspring[mask] = parents[0][mask]
        return offspring
    
    # uniform
    def __mutation(self, offspring):
        mutationrate = 0.9
        rng = self.seed
        size=np.shape(offspring)
        mutation = rng.uniform(size = size)
        mask = rng.choice([True, False], size = size, p = [mutationrate, \
                          1-mutationrate])
        offspring[mask] = mutation[mask]
        return offspring
        
    def __reproduce(self, parents):
        return self.__mutation(self.__crossover(parents))
    
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
        
        # Generar la primera generacion alpha : P(1)= [alfa1, alfa2]
        parents = np.random.uniform(0, self.C, size=(self.popsize, numVars))  
        
        # Calcular fitness
        fparents = np.asarray([self.__fitness(ind) for ind in parents])
        cparents = np.asarray([self.__constraint(ind) for ind in parents])
        
        currentGeneration = 0
        # Usando diferential evolution
        while currentGeneration < self.generations + 1:
            print ('GENERATION:', currentGeneration)
            # Para cada elemento de la poblacion
            for j in range(self.popsize):
                #--- MUTATION  ---------------------+
                idxs = [idx for idx in range(self.popsize) if idx != j]
                
                # Dividir el vector en tres random
                a, b, c = parents[np.random.choice(idxs, 3, replace = False)]
                print (np.shape(a))
                print (np.shape(b))
                print (np.shape(c))
                mutant = np.clip(a + self.mutationRate * (b - c), 0, 1)
                cross_points = np.random.rand(numVars) < self.crossoverRate
                
                # ---------- CROSSOVER --------------
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, numVars)] = True
                    
                # Seleccionar el hijo de la mutacion
                offspring = np.where(cross_points, mutant, parents[j])
                
                # Obtener el fitness y el contraint del hijo
                foffspring = self.__fitness(offspring)
                coffspring = self.__constraint(offspring)

                # --- SELECTION ----------------
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
    

    # Predecir los nuevos valores a partir de las alfas generadas    
    def predict(self, Xtest):
        
        # Xi = train
        # X =  test
        alphaY = np.multiply( self.solution.T, self.Ytrain)
        kernel = self.__compute_kernel(Xtest, self.Xtrain)
        yHat = np.dot(kernel, alphaY)
    
        # Calcaular el sesgo de la prediccion a partir de los vectores
        b = (np.min([yHat[i] for i in range(np.shape(yHat)[0]) if self.Ytrain[i] ==  1 and self.solution[i] > 0]) + np.max([yHat[i] for i in range(np.shape(yHat)[0]) if self.Ytrain[i] == -1 and self.solution[i] > 0]))/2
       
        # Llenar predict con los signos de yhat + b
        self.predict = np.sign(yHat + b)
        print ('predict', self.predict)
        #Terminando la optimizacion

        return self.predict
    
    # Funcion para calcular la presicion del algoritmo
    def accuracy(self, Ytest):
        # Contar los elementos que sean iguales en la prediccion y en Ytrain
        count = 0
        for i in range(np.shape(Ytest)[0]):
            if self.Ytrain[i] == self.predict[i]:
                count += 1
                
        
        # Calcular el promedio
        acc = count/Ytest.shape[0]
        print('Acc:', acc)
        return acc
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        return None
    
# Recibir los datasets
Xtrain = genfromtxt(fname ='./datasetCI2019/echocardiogram/Xtrain.csv', delimiter=',', dtype=None)
Ytrain = genfromtxt(fname ='./datasetCI2019/echocardiogram/Ytrain.csv', delimiter=',', dtype=None)
Xtest  = genfromtxt(fname ='./datasetCI2019/echocardiogram/Xtest.csv', delimiter=',', dtype=None)
Ytest  = genfromtxt(fname ='./datasetCI2019/echocardiogram/Xtest.csv', delimiter=',', dtype=None)

# Mandar a llamar al AE
sol = EvolutionaryConstrainedSVM( Xtrain, Ytrain)
sol.train()
sol.predict(Xtest)
sol.accuracy(Ytest)

