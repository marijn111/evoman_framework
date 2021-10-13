import os
import pickle
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
from ass1_controller import PlayerController
import matplotlib.pyplot as plt
import neat
import pandas as pd
import seaborn as sns

sns.set(font_scale=1.2)

enemy_groups = [[1, 2, 3], [3, 4, 5]]
neat_config_filename = 'neat_config.txt'
num_experiments = 10
num_best_solution_runs = 5

os.environ["SDL_VIDEODRIVER"] = "dummy"

config = neat.Config(
    genome_type=neat.DefaultGenome,
    reproduction_type=neat.DefaultReproduction,
    species_set_type=neat.DefaultSpeciesSet,
    stagnation_type=neat.DefaultStagnation,
    filename=neat_config_filename
)

if __name__ == "__main__":
    experiment_name = 'ass2-individual-gain'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    df = pd.DataFrame(columns=['enemy', 'algo', 'gain'])

    for e, enemy_group in enumerate(enemy_groups):
        for i in range(num_experiments):
            total_gain_neat = 0
            genome_neat = pickle.load(open('winner_objects/neat-winner_{}_{}.pickle'.format(e, i), 'rb'))
            ffn = neat.nn.FeedForwardNetwork.create(genome_neat, config)

            for j in range(num_best_solution_runs):
                env_neat = Environment(experiment_name=experiment_name,
                                       enemies=enemy_group,
                                       player_controller=PlayerController())

                _, player_life, enemy_life, _ = env_neat.play(pcont=ffn)
                total_gain_neat += player_life - enemy_life

            mean_gain = total_gain_neat / num_best_solution_runs
            df = df.append([{
                'group': e,
                'algo': 'neat',
                'gain': mean_gain
            }
            ], ignore_index=True)

    for e, enemy_group in enumerate(enemy_groups):
        for i in range(num_experiments):
            total_gain_neuro = 0
            genome_neuro = pickle.load(open('neuro_winners/generalist-winner_{}_{}.pickle'.format(e, i), 'rb'))

            for enem in range(1, 9):
                mean_total_gain = 0
                for j in range(num_best_solution_runs):
                    env_neuro = Environment(
                        experiment_name=experiment_name,
                        enemies=[enem],
                        player_controller=player_controller(10))

                    _, player_life, enemy_life, _ = env_neuro.play(pcont=genome_neuro)
                    mean_total_gain += player_life - enemy_life

                total_gain_neuro += mean_total_gain / num_best_solution_runs
            mean_gain = total_gain_neuro / num_best_solution_runs

            df = df.append([{
                'group': e,
                'algo': 'neuro',
                'gain': mean_gain
            }
            ], ignore_index=True)

    df.to_csv('ass2_gains_winners.csv', index=False)

    sns.boxplot(data=df, x='group', y='gain', hue='algo').set_title('Mean Gain of Winner Genomes')
    sns.despine(offset=10, trim=True)

    plt.tight_layout()
    plt.savefig('gains_winners_duo.png')
