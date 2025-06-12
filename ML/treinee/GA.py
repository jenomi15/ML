
    
import math   
import random
from matplotlib.pylab import choice


class Individual :
    def __init__(self, chromossome = None , index = 0 , fitness = None):
        self.chromossome = chromossome
        self.index = index
        self.fitness = 0 
        
    def toString(self):
        string = 'individuo' + str(self.index) + 'chromossome = ' + str(self.chromossome) + ", fitness score = " + str(self.fitness) 
        return string
      
class genetic_algorithm:
    
   def factorial(v):
    if v == 0:
        return 1
    else:
        return v * genetic_algorithm.factorial(v - 1)

       
       
   def __init__(self,max_generation = 10  ,size_population = 10  ,mutation_rate = 10 ):
        self.max_generation = max_generation
        self.size_population = size_population
        self.mutation_rate = mutation_rate
        self.highestindividual = Individual()
        self.chromotype = {
            "v" : range(1,10),
            "w" : range(1,20),
            "y" : range(1,5),
            "x" : range(10 ,30),
            "z" : range(1,30)   
        }
        self.population = []
        
   def generate_population(self,number):
       population = []
       for i in range(number):
            individual = Individual(index=i)
            individual.chromossome = {
                "v": random.choice(list(self.chromotype["v"])),
                "w": random.choice(list(self.chromotype["w"])),
                "y": random.choice(list(self.chromotype["y"])),
                "x": random.choice(list(self.chromotype["x"])),
                "z": random.choice(list(self.chromotype["z"]))
            }
            population.append(individual)

       for individuals in population:
            print(individuals.chromossome)
       self.population = population
       return population
   def fitnessscore(self, individual):
        c = individual.chromossome
        return (((genetic_algorithm.factorial(c["v"]) / 100) * c["w"]) 
                / (c["y"] ** (1 / c["x"]))) - (c["z"] ** (math.e - 2)) 
   def give_score(self):
       for individual in self.population:
           score = self.fitnessscore(individual)
           individual.fitness = score
       for individual in self.population:
            print(individual.fitness)
            
   def tournment_selection():
        print("oi")
    
   
   
        
        
                
            
                
            
            
            
            
            
            

        

        
ga = genetic_algorithm()
ga.generate_population(5)
ga.give_score()
    
    
    
    
            
        
        
     