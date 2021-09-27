# imports framework
import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import pandas as pd

# imports other libs
import numpy as np

os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'neuro_algorithm'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

num_experiments = 10
enemies = [1, 2, 3]

df = pd.DataFrame(columns=['fitness', 'type', 'enemy', 'gen', 'algo'])
enemy_number = 0
gen_iteration = 1

# Update the number of neurons for this specific example
n_hidden_neurons = 0

for enemy in enemies:
    enemy_number = enemy

    # initializes environment for single objective mode (specialist)  with static enemy and ai player
    env = Environment(
        experiment_name=experiment_name,
        playermode="ai",
        player_controller=player_controller(n_hidden_neurons),
        level=2
    )

    for i in range(num_experiments):
        gen_iteration = 1





