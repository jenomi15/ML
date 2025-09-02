
    
import math   
import random 
from random import choice



class Individual :
    def __init__(self, chromossome = None , index = 0 , fitness = None):
        self.chromossome = chromossome
        self.index = index
        self.fitness = 0 
        
    def __str__(self):
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
        self.parents = []
        
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
            
   def tournment_selection(self,population):
        parents = []
        tamanho = 2
        for _ in range(tamanho):
         
            pai1 = choice(population)
            pai2 = choice(population)
            while(pai1 == pai2):
                pai2 = choice(population)
            
            pai = pai1
            if pai1.fitness < pai2.fitness :
                pai = pai2
            parents.append(pai)
            self.parents = parents
        if parents[0] == parents[1]:
            return self.tournment_selection(population)

        for i in parents:
            print(i)
        return parents
              
             
        
   def crossover(self, parents):
       son = {}
       keys = self.chromotype.keys()
       for key in keys:
           son[key] = random.choice([parent.chromossome[key]for parent in parents])
           
       print(f" the son is : {son}")  
       return son
   
   def mutation(self,sons):
       keys = self.chromotype.keys()
       print(sons)
       for son in sons:
           for key in keys:
               if random.random() < 0.1:
                   son[key] = random.choice(list(self.chromotype[key]))
       print(sons)
       return sons
           

      
                   
     
           
         

        
ga = genetic_algorithm()
rodagens = int(input("digite a quantidade de vezes para o ga rodar "))
listofresults = []

for i in range(rodagens):
    population = ga.generate_population(5)
    ga.give_score()
    parents = ga.tournment_selection(population)
    son_chromossome = ga.crossover(parents)
    sons_chromossomes = ga.mutation([son_chromossome])
    
    for chromossome in sons_chromossomes:
        son = Individual(chromossome=chromossome, index=i)
        son.fitness = ga.fitnessscore(son)
        listofresults.append(son)

maior = listofresults[0]
for ind in listofresults:
    if ind.fitness > maior.fitness:
        maior = ind

print("O melhor indivíduo foi:")
print(maior)

        

    
    

    
    
    
    
            
        
        
     