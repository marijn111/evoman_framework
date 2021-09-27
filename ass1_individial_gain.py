import os
import pickle
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from ass1_controller import PlayerController
from demo_controller import player_controller
import matplotlib.pyplot as plt
import neat
import pandas as pd
import seaborn as sns

enemies = [1, 2, 3]
neat_config_filename = 'neat_config.txt'
num_experiments = 10
num_best_solution_runs = 5

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

    df = pd.DataFrame(columns=['enemy', 'EA', 'gain'])

    determine_winners = True
    winners_neat = [(0, -990), (0, -999)]
    winners_neuro = [(0, -999), (0, -999)]

    for enemy in enemies:
        if determine_winners:
            for i in range(num_experiments):
                # genome_neuro = pickle.load(open('winner_objects/neat-winner_{}_{}'.format(enemy, i), 'rb'))
                genome_neat = pickle.load(open('winner_objects/neat-winner_{}_{}.pickle'.format(enemy, i), 'rb'))
                ffn = neat.nn.FeedForwardNetwork.create(genome_neat, config)

                total_gain_neat = 0
                # total_gain_neuro = 0
                for e in enemies:
                    env_neat = Environment(
                        experiment_name=experiment_name,
                        enemies=[e],
                        player_controller=PlayerController()
                    )

                    _, pe_neat, ee_neat, _ = env_neat.play(pcont=ffn)
                    total_gain_neat += pe_neat - ee_neat

                    # env_neuro = Environment(experiment_name=experiment_name,
                    #                         enemies=[e],
                    #                         player_controller=player_controller(10))
                    #
                    # _, pe_neuro, ee_neuro, _ = env_neuro.play(pcont=genome_neuro)
                    # total_gain_neuro += pe_neuro - ee_neuro

                if winners_neat[enemy][1] < total_gain_neat:
                    winners_neat[enemy] = (i, total_gain_neat)
                # if winners_neuro[enemy][1] < total_gain_neuro:
                #     winners_neuro[enemy] = (i, total_gain_neuro)

        print(winners_neat)
        # print(winners_neuro)

        # genome_neuro = pickle.load(open('winner_objects/neat-winner_{}_{}'.format(enemy, winners_neuro[enemy][0]), 'rb'))
        genome_neat = pickle.load(open('winner_objects/neuro-winner_{}_{}'.format(enemy, winners_neat[enemy][0]), 'rb'))
        ffn = neat.nn.FeedForwardNetwork.create(genome_neat, config)

        for v in range(num_best_solution_runs):
            total_gain_neat = 0
            # total_gain_neuro = 0
            for enem in enemies:
                env_neat = Environment(experiment_name=experiment_name,
                                       enemies=[enem],
                                       player_controller=PlayerController())

                _, pe_neat, ee_neat, _ = env_neat.play(pcont=ffn)
                total_gain_neat += pe_neat - ee_neat

                # env_neuro = Environment(experiment_name=experiment_name,
                #                         enemies=[enem],
                #                         player_controller=player_controller(10))
                #
                # _, pe_neuro, ee_neuro, _ = env_neuro.play(pcont=genome_neuro)
                # total_gain_neuro += pe_neuro - ee_neuro

            df = df.append([{
                'enemy': enemy,
                'EA': 'neat',
                'gain': total_gain_neat
            }
                # , {
            #     'enemy': enemy,
            #     'EA': 'neuro',
            #     'gain': total_gain_neuro
            # }
            ], ignore_index=True)

    print(winners_neat)
    # print(winners_neuro)
    df.to_csv('results-winner-gains.csv', index=False)

    sns.boxplot(data=df, x='enemy', y='gain', hue='EA').set_title('gain of best solutions')
    sns.despine(offset=10, trim=True)

    plt.tight_layout()
    plt.savefig('gains.png')
