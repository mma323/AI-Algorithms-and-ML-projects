Implementation of a genetic algorithm for optimizing a population based on each individuals
distance to a food item.

Pygame is used for visualizing, and needs to be installed in running environment for code to work.
Main.py contains the render code. The initial population is also created here. 

GAengine.py will contains the bulk of the genetic algorithm implementation. 
Utils.py contains helper classes. GAPoint is a simple class representing a 
position in the world, with a function to find the distance to another GAPoint. 

Chromosome represents the chromosome and has skeleton functions for crossover 
and mutation. The chromosome only has two features, the x and y position of the 
individual in the world, used for rendering the individual in the simulator. 

