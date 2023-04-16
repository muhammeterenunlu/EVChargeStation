import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
from deap import base, creator, tools, algorithms

class NSGA2Optimizer:

    def __init__(self):
        pass

    def optimize_charging_stations(self, utility_cost_data, num_nodes, total_charging_stations, min_charging_stations, max_charging_stations):
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
        population_size = 50
        generations = 200
        crossover_probability = 0.9
        mutation_probability = 0.1

        # Initialize population
        pop = toolbox.population(n=population_size)

        # Run the NSGA-II algorithm
        pop, _, gen_history = self.custom_eaMuPlusLambda(pop, toolbox, mu=population_size, lambda_=population_size, cxpb=crossover_probability, mutpb=mutation_probability, ngen=generations, verbose=False)
        
        # Get the best solution
        best_solution = tools.selBest(pop, 1)[0]

        # Return the best solution found by NSGA-II
        return best_solution, gen_history
        
    def visualize_charging_stations(self, utility_cost_data, best_solution):
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

    def save_best_solution(self, generate_graph, best_solution, filename='best_solution.json'):
        best_solution_dict = {
            'Total Number of Installed Charging Stations': sum(best_solution),
            'Total Utility': sum(generate_graph.utility_cost_data[i]['Total Utility of 1 CS'] * best_solution[i] for i in range(generate_graph.num_nodes)),
            'Total Cost': int(sum(generate_graph.utility_cost_data[i]['Total Cost of 1 CS'] * best_solution[i] for i in range(generate_graph.num_nodes))),
            'Charging Stations': {f'Node {i}': best_solution[i] for i in range(generate_graph.num_nodes)}
        }
        with open(filename, 'w') as f:
            json.dump(best_solution_dict, f, indent=4)

    def custom_eaMuPlusLambda(self, population, toolbox, mu, lambda_, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Begin the generational process
        gen_history = []
        for gen in range(1, ngen + 1):
            # Select the next generation individuals
            offspring = toolbox.select(population, lambda_)

            # Vary the pool of individuals
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the old population by the offspring
            population[:mu] = offspring

            # Append the current generation to gen_history
            gen_info = {
                'generation': gen,
                'population': [ind[:] for ind in population],
                'fitnesses': [ind.fitness.values for ind in population]
            }
            gen_history.append(gen_info)

            # Update the statistics with the new population
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

        return population, logbook, gen_history

    def save_generation_history(self, gen_history, filename='generation_history.json'):
        gen_history_dict = []
        for gen_info in gen_history:
            feasible_population = []
            feasible_fitnesses = []
            for ind, fitness in zip(gen_info['population'], gen_info['fitnesses']):
                if fitness[0] != -1e9 and fitness[1] != -1e9:
                    feasible_population.append(ind)
                    feasible_fitnesses.append(fitness)
            gen_history_dict.append({
                'generation': gen_info['generation'],
                'population': [" ".join(str(x) for x in ind) for ind in feasible_population],
                'fitnesses': [list(fitness) for fitness in feasible_fitnesses]})
        with open(filename, 'w') as f:
            json.dump(gen_history_dict, f, indent=4)

    def visualize_pareto_front(self, gen_history, best_solution):
        # Get the last generation information
        last_generation = gen_history[-1]

        # Get the feasible solutions and their fitnesses from the last generation
        feasible_population = []
        feasible_fitnesses = []
        best_solution_fitness = None
        for ind, fitness in zip(last_generation['population'], last_generation['fitnesses']):
            if fitness[0] != -1e9 and fitness[1] != -1e9:
                feasible_population.append(ind)
                feasible_fitnesses.append(fitness)
                if ind == best_solution:
                    best_solution_fitness = fitness

        # Plot the feasible solutions
        plt.figure()
        plt.scatter(*zip(*feasible_fitnesses), marker='o', s=30, edgecolor='k')
        
        # Plot the best solution with a different color, for example, red
        if best_solution_fitness is not None:
            plt.scatter(best_solution_fitness[0], best_solution_fitness[1], marker='o', s=30, color='red', edgecolor='k')

        plt.xlabel("Total Utility")
        plt.ylabel("Total Cost")
        plt.title("Pareto Front")
        plt.savefig('pareto_front.png', bbox_inches='tight', dpi=300)
        plt.show()

    def run_optimization(self, generate_graph):
        # Set the total number of charging stations that can be installed
        total_charging_stations = round(generate_graph.num_nodes * 10)
        print(f"Maximum number of charging stations that can be installed: {total_charging_stations}")

        # Set the minimum and maximum number of charging stations based on the total number of charging stations
        min_charging_stations = 1
        max_charging_stations = round((total_charging_stations / generate_graph.num_nodes) * 2)
        
        # Run NSGA-II
        best_solution, gen_history = self.optimize_charging_stations(generate_graph.utility_cost_data, generate_graph.num_nodes, total_charging_stations, min_charging_stations, max_charging_stations)

        # Visualize the Pareto front
        self.visualize_pareto_front(gen_history, best_solution)

        # Save the generation history to a file in JSON format
        self.save_generation_history(gen_history)

        # Visualize the best solution found by NSGA-II
        self.visualize_charging_stations(generate_graph.utility_cost_data, best_solution)

        print(f"Total number of charging stations that are installed: {sum(best_solution)}")

        # Save the best solution to a file in JSON format
        self.save_best_solution(generate_graph, best_solution)