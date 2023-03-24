import random
from deap import base, creator, tools, algorithms

def optimize_charging_stations(utility_cost_data, num_nodes, total_charging_stations, min_charging_stations, max_charging_stations):
    # Objective functions
    def total_utility(individual):
        return sum(utility_cost_data[i]['Utility'] * individual[i] for i in range(num_nodes))

    def total_cost(individual):
        return sum(utility_cost_data[i]['Total Cost'] * individual[i] for i in range(num_nodes))

    # Constraint function
    def total_charging_stations_constraint(individual):
        return sum(individual) <= total_charging_stations

    # Create types
    # Indicate that the first objective (total utility) should be maximized (positive weight) while the second objective (total cost) should be minimized (negative weight).
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    # Initialize toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, min_charging_stations, max_charging_stations)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, num_nodes)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Define evaluation and selection functions
    def evaluate(individual):
        return total_utility(individual), total_cost(individual)

    def feasible(individual):
        return total_charging_stations_constraint(individual)

    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutUniformInt, low=min_charging_stations, up=max_charging_stations, indpb=0.1)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", evaluate)
    toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, (-1e9, -1e9)))

    # Set parameters for the NSGA-II algorithm
    population_size = 100
    generations = 100
    crossover_probability = 0.9
    mutation_probability = 0.1

    # Initialize population
    pop = toolbox.population(n=population_size)

    # Run the NSGA-II algorithm
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=population_size, lambda_=population_size, cxpb=crossover_probability, mutpb=mutation_probability, ngen=generations, verbose=False)

    # Get the best solution
    best_solution = tools.selBest(pop, 1)[0]

    # Return the best solution and corresponding utility and cost
    return best_solution, total_utility(best_solution), total_cost(best_solution)