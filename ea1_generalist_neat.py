################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
import neat
sys.path.insert(0, 'evoman')
from environment import Environment
from ass1_controller import PlayerController
import pickle
import pandas as pd

num_experiments = 10
enemies = [[1, 2, 3], [4,5,6]]
neat_config_filename = 'neat_generalist_config.txt'

# choose this for not using visuals and thus making experiments faster
os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'neat_generalist_algorithm'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


config = neat.Config(
    genome_type=neat.DefaultGenome,
    reproduction_type=neat.DefaultReproduction,
    species_set_type=neat.DefaultSpeciesSet,
    stagnation_type=neat.DefaultStagnation,
    filename=neat_config_filename
)

df = pd.DataFrame(columns=['fitness', 'type', 'group', 'gen', 'algo'])
group_number = 0
gen_iteration = 1


def eval_genomes(genomes, config):
    global df, gen_iteration
    fitness_values = []
    for genome_id, genome in genomes:
        neat_network = neat.nn.FeedForwardNetwork.create(genome, config)

        # f = fitness, p = player life, e = enemy life, t = game run time
        fitness, player_life, enemy_life, time = env.play(pcont=neat_network)
        fitness_values.append(fitness)
        genome.fitness = fitness

    max_fitness = max(fitness_values)
    mean_fitness = sum(fitness_values)/len(fitness_values)
    df = df.append({'fitness': max_fitness, 'type': 'max_fitness', 'group': group_number, 'gen': gen_iteration, 'algo': 'neat'}, ignore_index=True)
    df = df.append({'fitness': mean_fitness, 'type': 'mean_fitness', 'group': group_number, 'gen': gen_iteration, 'algo': 'neat'}, ignore_index=True)
    gen_iteration += 1


if __name__ == "__main__":

    for e, group in enumerate(enemies):
        group_number = e

        env = Environment(
            experiment_name=experiment_name,
            enemies=group,
            player_controller=PlayerController(),
            multiplemode='yes'
        )

        for i in range(num_experiments):
            # reset the gen iteration
            gen_iteration = 1
            # Create the population, which is the top-level object for a NEAT run.
            p = neat.Population(config)

            # Add a stdout reporter to show progress in the terminal.
            p.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            p.add_reporter(stats)
            p.add_reporter(neat.Checkpointer(5))

            winner = p.run(eval_genomes, 15)
            pickle.dump(winner, open('neat-winners/generalist-winner_{}_{}.pickle'.format(e, i), 'wb'))


df.to_csv('neat_generalist.csv', index=False)

