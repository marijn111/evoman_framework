import os
import pickle
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns

sys.path.insert(0, 'evoman')
from evoman.environment import Environment
from demo_controller import player_controller

sns.set(font_scale=1.2)
mpl.rcParams['figure.dpi'] = 300

experiment_name = 'test-best'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


enemies = [[1, 2, 3], [4,5,6]]
df = pd.DataFrame(columns=['group', 'algo', 'gain'])
determine_winners = True
winners_neuro = [(0, -999), (0, -999)]
num_experiments = 10
num_repetitions = 5
num_hidden_neurons = 10

if __name__ == "__main__":

    for e, enemy_group in enumerate(enemies):
        print("e:", enemy_group)
        if determine_winners:
            for r in range(num_experiments):
                genome_neuro = pickle.load(open('neuro_winners/generalist-winner_{}_{}.pickle'.format(e, r), 'rb'))
                total_gain_neuro = 0
                for enemy in range(1, 9):
                    env_neuro = Environment(experiment_name=experiment_name,
                                            enemies=[enemy],
                                            player_controller=player_controller(num_hidden_neurons))

                    _, pe_neuro, ee_neuro, _ = env_neuro.play(pcont=genome_neuro)
                    total_gain_neuro += pe_neuro - ee_neuro

                if winners_neuro[e][1] < total_gain_neuro:
                    winners_neuro[e] = (r, total_gain_neuro)

        print(winners_neuro)

        genome_neuro = pickle.load(open('neuro_winner/neuro-winner-e{}-r{}'.format(e, winners_neuro[e][0]), 'rb'))

        avg_energy_table = [(0,0) for i in range(1, 9)]
        for v in range(num_repetitions):
            print("v:", v)
            total_gain_neuro = 0
            for enemy in range(1, 9):
                env_neuro = Environment(experiment_name=experiment_name,
                                        enemies=[enemy],
                                        player_controller=player_controller(num_hidden_neurons))

                _, pe_neuro, ee_neuro, _ = env_neuro.play(pcont=genome_neuro)
                total_gain_neuro += pe_neuro - ee_neuro

                avg_energy_table[enemy][0] += pe_neuro
                avg_energy_table[enemy][1] += ee_neuro

            df = df.append([{
                'group': e,
                'algo': 'neuro',
                'gain': total_gain_neuro
            }], ignore_index=True)

        for e, enem in enumerate(avg_energy_table):
            print(f'Enemy {e}')
            print(f'Player energy: {enem[0]/num_repetitions}')
            print(f'Enemy Energy: {enem[1]/num_repetitions}')


    print(winners_neuro)
    df.to_csv('ass2_neuro_gain.csv', index=False)

    # sns.boxplot(data=df, x='enemy', y='gain', hue='EA').set_title('gain of best solutions')
    # sns.despine(offset=10, trim=True)

    # plt.tight_layout()
    # plt.savefig('gains.png')