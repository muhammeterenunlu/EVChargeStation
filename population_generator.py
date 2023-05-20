import random
from deap import creator, tools, base
from rand_seed_creator import rand_seed_creator

creator.create("FitnessMultiObjective", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMultiObjective)

def generate_population(num_nodes, min_charging_stations, max_charging_stations, population_size, seed=rand_seed_creator):
    random.seed(seed)  # Set the seed
    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, min_charging_stations, max_charging_stations)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, num_nodes)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    pop = toolbox.population(n=population_size)
    """ For debugging
    print("Population generated")
    # print population
    print(pop)
    """
    return pop
