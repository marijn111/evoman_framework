import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, 'evoman')
from evoman.environment import Environment
from demo_controller import player_controller

experiment_name = 'neuro_evolution'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

os.environ["SDL_VIDEODRIVER"] = "dummy"

rng = np.random.default_rng()


df = pd.DataFrame(columns=['fitness', 'type', 'group', 'gen', 'algo'])

# parameters
enemy_groups = [[1, 2, 3], [3, 4, 5]]
number_of_runs = 10
number_hidden_nodes = 10
max_gen = 10
max_stag = 10
pop_size = 20
surv_rate = 0.3
number_elites = 1
init_species = 10
min_species = 3
min_species_size = 2
mut_parameters = {
    "mut_rate": 0.5,
    "mut_power": 0.7,
    "mut_replace_rate": 0.25
}

# The agents have access to 20 sensors
num_inputs = 20
# The player has 5 different possible outcomes: left, right, jump, shoot, release
num_outputs = 5


def init_population(pop_size, number_inputs, number_ouputs, number_hidden_nodes, init_species):
    indiv_length = (number_inputs + 1) * number_hidden_nodes + number_ouputs * (number_hidden_nodes + 1)
    pop = rng.normal(size=(pop_size, indiv_length))
    return np.array_split(pop, init_species)


def get_fitness(indiv_index, species_index, indiv, env):
    fitness, _, _, _ = env.play(pcont=indiv)
    result = {
        "indiv_index": indiv_index,
        "species_index": species_index,
        "fitness": fitness
    }
    print(result)
    return result


def generate_children(parents, number_children):
    children = []

    for _ in range(number_children):
        parent_one, parent_two = rng.choice(parents, size=2, replace=False)
        child = uniform_crossover(parent_one, parent_two)
        mutate(child)
        children.append(child)

    return children


def uniform_crossover(parent_one, parent_two):
    prop = rng.random(len(parent_one))
    child = parent_one * prop + parent_two * (1 - prop)
    return child


def mutate(indiv):
    global mut_parameters
    for i in range(len(indiv)):
        if rng.random() < mut_parameters["mut_replace_rate"]:
            indiv[i] = rng.normal()
        else:
            if rng.random() < mut_parameters["mut_rate"]:
                indiv[i] = rng.normal(loc=indiv[i], scale=mut_parameters["mut_power"])
    return indiv


if __name__ == "__main__":
    for e, enemies in enumerate(enemy_groups):
        env = Environment(experiment_name=experiment_name,
                          enemies=enemies,
                          multiplemode='yes',
                          player_controller=player_controller(number_hidden_nodes))

        for r in range(number_of_runs):
            pop = init_population(pop_size, num_inputs, num_outputs, number_hidden_nodes, init_species)
            number_species = init_species
            stag_array = [[0, -99] for _ in range(number_species)]
            winner = [None, 0]

            for g in range(max_gen):

                print(f"generation {g}")

                gen_results = []
                elites = []
                parents = []
                max_fitnesses = np.zeros(number_species)

                for s, sub_pop in enumerate(pop):
                    species_gen_results = [get_fitness(i, s, indiv, env) for i, indiv in
                                           enumerate(sub_pop)]
                    gen_results.append(species_gen_results)

                    sorted_results = sorted(
                        species_gen_results,
                        key=lambda result: result["fitness"],
                        reverse=True
                    )

                    species_max_fitness = sorted_results[0]["fitness"]
                    max_fitnesses[s] = species_max_fitness
                    if species_max_fitness > winner[1]:
                        winner = [sub_pop[sorted_results[0]["indiv_index"]], species_max_fitness]

                    species_elites = [sub_pop[result["indiv_index"]]
                                      for result in sorted_results[:number_elites]]
                    elites.append(species_elites)

                    number_survivors = max(int(len(sub_pop) * surv_rate), min_species_size)
                    number_children = len(sub_pop) - number_elites
                    species_parents = [sub_pop[result["indiv_index"]]
                                       for result in sorted_results[:number_survivors]]
                    parents.append(species_parents)

                max_fitness = max(max_fitnesses)
                sum_fitnesses = 0
                for i in range(number_species):
                    sum_species_fitness = 0
                    for j in range(len(pop[i])):
                        fitness = gen_results[i][j]["fitness"]
                        sum_fitnesses += fitness
                        sum_species_fitness += fitness

                mean_fitness = sum_fitnesses / sum([len(sp) for sp in pop])
                print("overal: max fit: {}, mean fit {}".format(max_fitness, mean_fitness))
                df = df.append({'fitness': max_fitness, 'type': 'max_fitness', 'group': e, 'gen': g, 'algo': 'neuro'}, ignore_index=True)
                df = df.append({'fitness': mean_fitness, 'type': 'mean_fitness', 'group': e, 'gen': g, 'algo': 'neuro'}, ignore_index=True)

                stag_species = []
                pop_lost = 0
                argsort = np.argsort(np.argsort(-max_fitnesses))
                for i in range(len(pop)):
                    if stag_array[i][1] < max_fitnesses[i]:
                        stag_array[i][1] = max_fitnesses[i]
                        stag_array[i][0] = 0
                    else:
                        stag_array[i][0] += 1
                        if stag_array[i][0] > max_stag and argsort[i] > min_species:
                            print(f"species {i} stagnated")
                            stag_species.append(i)
                            pop_lost += len(pop[i])
                            number_species -= 1

                for i in reversed(stag_species):
                    pop.pop(i)
                    stag_array.pop(i)
                    elites.pop(i)
                    parents.pop(i)
                    max_fitnesses = np.delete(max_fitnesses, i)

                for _ in range(pop_lost):
                    max_fitnesses_shift = max_fitnesses - min(
                        max_fitnesses)
                    p = max_fitnesses_shift / sum(max_fitnesses_shift)
                    sub_pop = rng.choice(pop, p=p)
                    np.vstack((sub_pop, sub_pop[
                        -1]))

                for i in range(number_species):
                    number_children = len(pop[i]) - number_elites
                    children = generate_children(parents[i], number_children)
                    pop[i] = elites[i] + children

            print("winner:", winner)
            pickle.dump(winner[0], open('neuro_winners/generalist-winner_{}_{}.pickle'.format(e, r), 'wb'))

    df.to_csv('neuro-generalist-results.csv', index=False)
