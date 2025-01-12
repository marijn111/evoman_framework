import os
import pickle
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from ea2_environment import CustomEnvironment
from ass1_controller import PlayerController
import matplotlib.pyplot as plt
import neat
import pandas as pd
import seaborn as sns

sns.set(font_scale=1.2)

enemies = [1, 2, 3]
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
    experiment_name = 'individual-gain'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    df = pd.DataFrame(columns=['enemy', 'algo', 'gain'])

    for enemy in enemies:
        for i in range(num_experiments):
            total_gain_neat = 0
            genome_neat = pickle.load(open('winner_objects/neat-winner_{}_{}.pickle'.format(enemy, i), 'rb'))
            ffn = neat.nn.FeedForwardNetwork.create(genome_neat, config)
            for j in range(num_best_solution_runs):
                env_neat = Environment(experiment_name=experiment_name,
                                       enemies=[enemy],
                                       player_controller=PlayerController())

                _, player_life, enemy_life, _ = env_neat.play(pcont=ffn)
                total_gain_neat += player_life - enemy_life

            mean_gain = total_gain_neat / num_best_solution_runs
            df = df.append([{
                'enemy': enemy,
                'algo': 'neat',
                'gain': mean_gain
            }
            ], ignore_index=True)

    for enemy in enemies:
        for i in range(num_experiments):
            total_gain_custom_neat = 0
            genome_neat = pickle.load(open('custom_winner_objects/neat-winner_{}_{}.pickle'.format(enemy, i), 'rb'))
            ffn = neat.nn.FeedForwardNetwork.create(genome_neat, config)
            for j in range(num_best_solution_runs):
                env_neat = CustomEnvironment(
                    experiment_name=experiment_name,
                    enemies=[enemy],
                    player_controller=PlayerController())

                _, player_life, enemy_life, _ = env_neat.play(pcont=ffn)
                total_gain_custom_neat += player_life - enemy_life

            mean_gain = total_gain_custom_neat / num_best_solution_runs
            df = df.append([{
                'enemy': enemy,
                'algo': 'custom_neat',
                'gain': mean_gain
            }
            ], ignore_index=True)

    df.to_csv('gains_winners_duo.csv', index=False)

    sns.boxplot(data=df, x='enemy', y='gain', hue='algo').set_title('Individual Gain of Winner Genomes')
    sns.despine(offset=10, trim=True)

    plt.tight_layout()
    plt.savefig('gains_winners_duo.png')
