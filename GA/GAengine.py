import random
from Utils import Chromosome, Food

class GAEngine:
    def __init__(self):
        self.population = []
        self.food = []
        self.generations = 0


    def make_initial_population(self, population_size):       
        for i in range(population_size):
            self.population.append(
                Chromosome(random.randint(0, 790), random.randint(0, 590))
            )


    def add_food(self, no_of_food):     
        for i in range(no_of_food):
            self.food.append(
                Food(random.randint(0, 790), random.randint(0, 590))
            )


    def do_crossover(self, no_of_offspring):
        """
        Creates new offspring by selecting the fittest individuals,
        and crossing them over to create new offspring.

        The fittest individuals are selected by sorting the population
        by fitness, and then selecting the fittest individuals.

        no_of_offspring is a hyperparameter that can be adjusted to
        find the optimal number of offspring for the problem.

        In this case elitist selection is used, due to its simplicity
        and reliability. 
        """

        self.population.sort(key=lambda x: x.get_fitness())

        parents = self.population[:no_of_offspring]

        for i in range(no_of_offspring-1):
            offspring = parents[i-1].crossover(parents[i+1])
            offspring.mutate()
            self.population.append(offspring)

        for individual in self.population:
            individual.set_fitness(0)

        self.generations += 1


    def assign_fitness(self):
        """
        Calculates the fitness of each individual in the population,
        based on their distance to the food.

        The fitness is set to the inverse of the distance to the food, 
        as the closer the individual is to the food, the higher the fitness.

        Fitness is only updated if the new fitness is higher than 
        the previous fitness. This way the distance to the CLOSEST food
        is the only distance that is used to calculate the fitness.
        """
        for individual in self.population:
            for food in self.food:
                fitness = 1 / individual.get_distance_to(food)
                if fitness > individual.get_fitness():
                    individual.set_fitness( fitness )


    def get_population(self):
        return self.population


    def get_foods(self):
        return self.food