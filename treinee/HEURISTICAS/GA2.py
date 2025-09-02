import math
import random
from random import choice, sample


class Individual:
    def __init__(self, chromossome=None, index=0):
        self.chromossome = chromossome
        self.index = index
        self.fitness = 0

    def __str__(self):
        string = f'Indivíduo {self.index} | Cromossomo = {self.chromossome} | Fitness = {self.fitness:.4f}'
        return string


class genetic_algorithm:

    @staticmethod
    def factorial(v):
        if v == 0:
            return 1
        else:
            return v * genetic_algorithm.factorial(v - 1)

    def __init__(self, max_generation=10, size_population=10, mutation_rate=0.1):
        self.max_generation = max_generation
        self.size_population = size_population
        self.mutation_rate = mutation_rate
        self.population = []
        self.chromotype = {
            "v": range(1, 10),
            "w": range(1, 20),
            "y": range(1, 5),
            "x": range(10, 30),
            "z": range(1, 30)
        }

    def generate_population(self):
        population = []
        for i in range(self.size_population):
            individual = Individual(index=i)
            individual.chromossome = {
                "v": random.choice(list(self.chromotype["v"])),
                "w": random.choice(list(self.chromotype["w"])),
                "y": random.choice(list(self.chromotype["y"])),
                "x": random.choice(list(self.chromotype["x"])),
                "z": random.choice(list(self.chromotype["z"]))
            }
            population.append(individual)

        self.population = population
        return population

    def fitnessscore(self, individual):
        c = individual.chromossome
    
        return (((genetic_algorithm.factorial(c["v"]) / 100) * c["w"])
                    / (c["y"] ** (1 / c["x"]))) - (c["z"] ** (math.e - 2))
        

    def evaluate_population(self):
        for individual in self.population:
            individual.fitness = self.fitnessscore(individual)

 
    def tournament_selection(self):    
        population_copy = self.population[:]
        parents = []
        while len(parents) < 2 and len(population_copy) >= 2:
          pai1, pai2 = sample(population_copy, 2)
          population_copy.remove(pai1)
          population_copy.remove(pai2)
          pai = pai1 if pai1.fitness > pai2.fitness else pai2
          if pai not in parents:
             parents.append(pai)
        return parents

    def crossover(self, parents):
        son = {}
        keys = self.chromotype.keys()
        for key in keys:
            son[key] = random.choice([parent.chromossome[key] for parent in parents])
        return son

    def mutation(self, chromossome):
        keys = self.chromotype.keys()
        for key in keys:
            if random.random() < self.mutation_rate:
                chromossome[key] = random.choice(list(self.chromotype[key]))
        return chromossome

    def evolve(self):
        self.generate_population()
        self.evaluate_population()

        for generation in range(self.max_generation):
            print(f"\n--- Geração {generation + 1} ---")
            new_population = []

            for i in range(self.size_population):
                parents = self.tournament_selection()
                child_chromossome = self.crossover(parents)
                mutated_chromossome = self.mutation(child_chromossome)

                child = Individual(chromossome=mutated_chromossome, index=i)
                child.fitness = self.fitnessscore(child)
                new_population.append(child)

            self.population = new_population
            self.evaluate_population()

            best = max(self.population, key=lambda ind: ind.fitness)
            print(f"Melhor da geração {generation + 1}: {best}")

        best_overall = max(self.population, key=lambda ind: ind.fitness)
        print("\n==== Melhor indivíduo encontrado ====")
        print(best_overall)
        return best_overall



if __name__ == "__main__":
    max_gen = int(input("Digite o número de gerações: "))
    pop_size = int(input("Digite o tamanho da população: "))

    ga = genetic_algorithm(max_generation=max_gen, size_population=pop_size, mutation_rate=0.1)

    best_individual = ga.evolve()

    
    
    