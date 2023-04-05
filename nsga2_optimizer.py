import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from deap import base, creator, tools, algorithms

def optimize_charging_stations(utility_cost_data, num_nodes, total_charging_stations, min_charging_stations, max_charging_stations):
    # Objective functions
    def total_utility(individual):
        return sum(utility_cost_data[i]['Total Utility of 1 CS'] * individual[i] for i in range(num_nodes))

    def total_cost(individual):
        return sum(utility_cost_data[i]['Total Cost of 1 CS'] * individual[i] for i in range(num_nodes))

    # Constraint function
    def total_charging_stations_constraint(individual):
        return sum(individual) <= total_charging_stations

    # Create types
    # Indicate that the first objective (total utility) should be maximized (positive weight) while the second objective (total cost) should be minimized (negative weight).
    creator.create("FitnessMultiObjective", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMultiObjective)

    # Initialize a DEAP toolbox object that will store various components required for the NSGA-II algorithm
    toolbox = base.Toolbox()

    # Register 'attr_int' as an integer attribute generator that generates random integers between min_charging_stations and max_charging_stations
    toolbox.register("attr_int", random.randint, min_charging_stations, max_charging_stations)

    # Register 'individual' as a function that creates an individual (solution) by repeatedly calling the 'attr_int' function num_nodes times
    # The resulting individual is a list of integers representing the number of charging stations at each node
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, num_nodes)

    # Register 'population' as a function that creates a list of individuals (a population) by repeatedly calling the 'individual' function
    # The number of individuals in the population will be specified when the function is called
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    # Define evaluation and selection functions
    def evaluate(individual):
        return total_utility(individual), total_cost(individual)

    def feasible(individual):
        return total_charging_stations_constraint(individual)

    # Register 'mate' as a function that performs uniform crossover on two individuals with the specified probability of exchanging each attribute (indpb)
    # In this case, the probability of exchanging each attribute between two individuals is 0.5 (50%)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)

    # Register 'mutate' as a function that performs uniform integer mutation on an individual with the specified mutation range and probability of mutating each attribute (indpb)
    # In this case, the mutation range is between min_charging_stations and max_charging_stations, and the probability of mutating each attribute is 0.1 (10%)
    toolbox.register("mutate", tools.mutUniformInt, low=min_charging_stations, up=max_charging_stations, indpb=0.1)

    # Register 'select' as a function that performs selection using the NSGA-II algorithm
    toolbox.register("select", tools.selNSGA2)

    # Register 'evaluate' as a function that calculates the objective values (total utility and total cost) of an individual
    toolbox.register("evaluate", evaluate)

    # Add a decorator to the 'evaluate' function to enforce the constraint that the total number of charging stations should not exceed the specified limit
    # The DeltaPenalty function applies a penalty to the objective values of infeasible individuals, making them less likely to be selected
    # In this case, a large penalty of (-1e9, -1e9) is applied to both objective values if the constraint is not satisfied
    toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, (-1e9, -1e9)))


    # Set parameters for the NSGA-II algorithm
    population_size = 100
    generations = 100
    crossover_probability = 0.9
    mutation_probability = 0.1

    # Initialize population
    pop = toolbox.population(n=population_size)

    # Run the NSGA-II algorithm
    algorithms.eaMuPlusLambda(pop, toolbox, mu=population_size, lambda_=population_size, cxpb=crossover_probability, mutpb=mutation_probability, ngen=generations, verbose=False)

    # Get the best solution
    best_solution = tools.selBest(pop, 1)[0]

    # Return the best solution found by NSGA-II
    return best_solution
    
def visualize_charging_stations(utility_cost_data, best_solution):
    node_indices = np.arange(len(utility_cost_data))
    charging_stations = [best_solution[i] for i in range(len(utility_cost_data))]

    plt.bar(node_indices, charging_stations)
    plt.xlabel('Node Index')
    plt.ylabel('Number of Charging Stations')
    plt.title('Charging Stations Distribution')
    
    # Set y-axis labels as integers
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=1))

    # Display x-tick labels every 5 nodes
    x_tick_labels = ['' if i % 10 != 0 else str(i) for i in node_indices]
    plt.xticks(node_indices, x_tick_labels, fontsize=8, rotation=0)

    # Save the image
    plt.savefig('charging_stations_distribution.png', bbox_inches='tight', dpi=300)
    plt.show()