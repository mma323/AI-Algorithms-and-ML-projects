import math, random


class GAPoint:
    def __init__(self, x_pos, y_pos):
        self.x_pos = x_pos
        self.y_pos = y_pos


    def get_distance_to(self, other):
        return math.sqrt(
            math.pow(
            self.x_pos - other.x_pos, 2) + math.pow(self.y_pos - other.y_pos
            , 2)
        )


class Chromosome(GAPoint):
    def __init__(self, x_pos, y_pos):
        self.fitness = 0
        super().__init__(x_pos, y_pos)


    def set_fitness(self, fitness):
        self.fitness = fitness


    def get_fitness(self):
        return self.fitness


    def crossover(self, other):
        """
        Creates a new chromosome by crossing over this chromosome with another chromosome.

        This is done using arithmetic crossover, where the new chromosome is the average
        of the two parents. Arithmetic crossover is used because it is a simple and reliable way to
        create a new chromosome that is a mix of the two parents.
        """

        self.x_pos = (self.x_pos + other.x_pos) / 2
        self.y_pos = (self.y_pos + other.y_pos) / 2
        return self


    # mutate the individual
    def mutate(self):
        """
        Mutates the chromosome by changing the x and y position randomly.

        The mutation rate determines the probability of a mutation happening,
        and can be adjusted to find the optimal mutation rate for the problem.

        The interval of the random number is determined by the problem, 
        in this case it was determined by looking at the interval in which 
        food can spawn. (See class Food)

        Notice that feature encoding is not used in this example, 
        as the featuresare already numerical x and y coordinates, 
        which makes the mutation easy to implement by just changing
        the coordinates randomly.
        """
        mutation_rate = 0.1

        if random.random() < mutation_rate:
            self.x_pos = random.randint(10, 790)
        if random.random() < mutation_rate:
            self.y_pos = random.randint(10, 590)


class Food(GAPoint):
    def __init__(self, x_pos, y_pos):
        self.amount = 100
        super().__init__(x_pos, y_pos)


    def reduce_amount(self):
        self.amount -= 1


    def get_amount(self):
        return self.amount


    def reposition(self):
        self.amount = 100
        self.x_pos = random.randint(10, 790)
        self.y_pos = random.randint(10, 590)



