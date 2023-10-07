import pygame, sys, time, random
from pygame.locals import *
from GAengine import GAEngine

simulator_speed = 50

red_color = pygame.Color(255, 0, 0)
green_color = pygame.Color(0, 255, 0)
blue_color = pygame.Color(0, 0, 255)
black_color = pygame.Color(0, 0, 0)
white_color = pygame.Color(255, 255, 255)

pygame.init()
fps_clock = pygame.time.Clock()

play_surface = pygame.display.set_mode((800, 600))
pygame.display.set_caption('ACO for pathfinding')

ga = GAEngine()
initial_population        : int = 200
food_amount               : int = 2
fittest_individuals       : int = 20
ga.make_initial_population(initial_population   )
ga.add_food(food_amount)

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                pygame.event.post(pygame.event.Event(QUIT))
                running = False

 
    ga.assign_fitness()
    ga.do_crossover(fittest_individuals)
    pygame.display.set_caption(
        'ACO for pathfinding - Generation: ' + str(ga.generations)
    )

    play_surface.fill(white_color)

    for pp in ga.get_population():
        pygame.draw.rect(
            play_surface,
            black_color, 
            Rect(pp.x_pos - 1, pp.y_pos - 1, 22, 22)
        )
        pygame.draw.rect(
            play_surface, 
            green_color, 
            Rect(pp.x_pos, pp.y_pos, 20, 20)
        )

    for food in ga.get_foods():
        food_size = food.get_amount() / 100 * 40
        pygame.draw.rect(
            play_surface, 
            red_color, 
            Rect(
                food.x_pos - food_size / 2, 
                food.y_pos - food_size / 2, 
                food_size, food_size
            )
        )

    pygame.display.flip()

    fps_clock.tick(simulator_speed)