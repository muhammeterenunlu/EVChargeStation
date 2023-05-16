import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
from collections.abc import Iterable
from deap import base, creator, tools, algorithms

class NSGA2Optimizer2:

    # Add a global variable to store the crossover information
    crossover_info = []

    def __init__(self,graph_generator):
        self.graph_generator = graph_generator

    # This method sets up and runs the NSGA-II algorithm, defining objective functions, constraints, crossover, mutation, and selection functions. 
    # It returns the best solution found and the generation history.
    def optimize_charging_stations(self, utility_cost_data, num_nodes, total_charging_stations, min_charging_stations, max_charging_stations):
        # Objective functions
        # Calculates the total utility of an individual (a solution) based on the number of charging stations at each node and the utility data.
        def total_utility(individual):
            total_utility_value = 0
            for i in range(num_nodes):
                utility_value = utility_cost_data[i]['Utility Specified By Using ML']
                total_utility_value += utility_value * individual[i]
            return total_utility_value

        # Calculates the total cost of an individual (a solution) based on the number of charging stations at each node and the cost data.
        def total_cost(individual):
            total_cost_value = 0
            for i in range(num_nodes):
                cost_value = utility_cost_data[i]['Cost Specified By Using ML']
                total_cost_value += cost_value * individual[i]
            return total_cost_value

        # Checks whether the total number of charging stations in an individual (a solution) does not exceed the specified limit.
        def total_charging_stations_constraint(individual):
            return sum(individual) <= total_charging_stations

        # Indicate that the first objective (total utility) should be maximized (positive weight) while the second objective (total cost) should be minimized (negative weight).
        # Inherited from the base.Fitness class
        # Check if the 'FitnessMultiObjective' class already exists before creating it
        if not hasattr(creator, 'FitnessMultiObjective'):
            creator.create("FitnessMultiObjective", base.Fitness, weights=(1.0, -1.0))

        # Indicate that the individuals (solutions) should be represented as lists.
        # Check if the 'Individual' class already exists before creating it
        if not hasattr(creator, 'Individual'):
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

        # Define a function that checks whether an individual (a solution) satisfies the constraint that the total number of charging stations should not exceed the specified limit
        def feasible(individual):
            return total_charging_stations_constraint(individual)

        # Register 'mate'(crossover) as a function that performs graph crossover on two individuals
        toolbox.register("mate", self.subgraph_crossover)

        # Register 'mutate' as a function that performs uniform integer mutation on an individual with the specified mutation range and probability of mutating each attribute (indpb)
        # In this case, the mutation range is between min_charging_stations and max_charging_stations, and the probability of mutating each attribute is 0.1 (10%)
        toolbox.register("mutate", tools.mutUniformInt, low=min_charging_stations, up=max_charging_stations, indpb=0)

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
        mutation_probability = 0

        # Initialize population
        pop = toolbox.population(n=population_size)

        # Run the NSGA-II algorithm
        pop, _, gen_history = self.custom_eaMuPlusLambda(pop, toolbox, mu=population_size, lambda_=population_size, cxpb=crossover_probability, mutpb=mutation_probability, ngen=generations, verbose=False)
        
        # Get the best solution
        best_solution = tools.selBest(pop, 1)[0]

        # Return the best solution found by NSGA-II
        return best_solution, gen_history
    
    def subgraph_crossover(self, ind1, ind2):
        # Get 10 subgraphs from each parent
        subgraphs_parent1 = [self.graph_generator.generate_connected_subgraph() for _ in range(10)]
        subgraphs_parent2 = [self.graph_generator.generate_connected_subgraph() for _ in range(10)]

        # Combine subgraphs from both parents
        all_subgraphs = subgraphs_parent1 + subgraphs_parent2

        # Determine the best subgraph based on total_utility/total_cost
        def subgraph_score(subgraph):
            total_utility = 0
            total_cost = 0
            for node in subgraph.nodes:
                total_utility += self.graph_generator.utility_cost_data[node]['Utility Specified By Using ML']
                total_cost += self.graph_generator.utility_cost_data[node]['Cost Specified By Using ML']
            return total_utility / total_cost

        best_subgraph = max(all_subgraphs, key=lambda subgraph: subgraph_score(subgraph[0]))

        # Perform crossover using the best subgraph
        subgraph_nodes = list(best_subgraph[0].nodes)
        temp1 = ind1[:]
        temp2 = ind2[:]
        for node in subgraph_nodes:
            ind1[node], ind2[node] = ind2[node], ind1[node]

        # Determine the parent of the selected subgraph
        selected_parent = None
        if best_subgraph in subgraphs_parent1:
            selected_parent = "Parent 1"
        elif best_subgraph in subgraphs_parent2:
            selected_parent = "Parent 2"

        crossover_info = self.crossover_info
        crossover_info.append({
            'parent1': temp1[:],
            'parent2': temp2[:],
            'offspring1': ind1[:],
            'offspring2': ind2[:],
            'subgraph_nodes': subgraph_nodes,
            'selected_node': best_subgraph[1],
            'selected_subgraph_parent': selected_parent
        })

        return ind1, ind2

      
    def visualize_charging_stations(self, utility_cost_data, best_solution):
        node_indices = np.arange(len(utility_cost_data))
        charging_stations = []
        for i in range(len(utility_cost_data)):
            charging_stations.append(best_solution[i])

        plt.bar(node_indices, charging_stations)
        plt.xlabel('Node Index')
        plt.ylabel('Number of Charging Stations')
        plt.title('Charging Stations Distribution')
        
        # Set y-axis labels as integers
        ax = plt.gca()
        ax.yaxis.set_major_locator(ticker.MultipleLocator(base=1))

        # Display x-tick labels every 5 nodes
        x_tick_labels = []
        for i in node_indices:
            if i % 10 != 0:
                x_tick_labels.append('')
            else:
                x_tick_labels.append(str(i))

        plt.xticks(node_indices, x_tick_labels, fontsize=8, rotation=0)

        # Save the image
        plt.savefig('crossover2_graph_figures_jsons/charging_stations_distribution.png', bbox_inches='tight', dpi=300)
        plt.show()

    def save_best_solution(self, best_solution):
        total_utility = 0
        total_cost = 0
        charging_stations = {}

        for i in range(self.graph_generator.num_nodes):
            total_utility += self.graph_generator.utility_cost_data[i]['Total Utility of 1 CS'] * best_solution[i]
            total_cost += self.graph_generator.utility_cost_data[i]['Total Cost of 1 CS'] * best_solution[i]
            charging_stations[f'Node {i}'] = best_solution[i]

        total_installed_charging_stations = sum(best_solution)

        best_solution_dict = {
            'Total Number of Installed Charging Stations': total_installed_charging_stations,
            'Total Utility': total_utility,
            'Total Cost': int(total_cost),
            'Charging Stations': charging_stations
        }
        with open('crossover2_graph_figures_jsons/best_solution.json', 'w') as f:
            json.dump(best_solution_dict, f, indent=4)

    # Custom implementation of the (μ + λ) evolutionary algorithm.
    def custom_eaMuPlusLambda(self, population, toolbox, mu, lambda_, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
        # Create a logbook to record statistics and information
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals']

        if stats:
            logbook.header.extend(stats.fields)

        # Evaluate the fitness of individuals with invalid fitness values in the initial population
        invalid_ind = []
        for ind in population:
            if not ind.fitness.valid:
                invalid_ind.append(ind)
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the initial population
        if halloffame is not None:
            halloffame.update(population)

        # Compile and record initial statistics
        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Initialize the generation history list
        gen_history = []

        # Main loop for the number of generations specified
        for gen in range(1, ngen + 1):
            # Select 'lambda_' individuals for the next generation
            offspring = toolbox.select(population, lambda_)

            # Apply crossover and mutation operations to the selected offspring
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

            # Evaluate the fitness of offspring with invalid fitness values
            invalid_ind = []
            for ind in offspring:
                if not ind.fitness.valid:
                    invalid_ind.append(ind)

            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the new offspring
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the 'mu' worst individuals in the population with the offspring
            population[:mu] = offspring

            # Store the current generation's information in gen_history
            population_copy = []
            fitnesses_copy = []

            for ind in offspring:
                population_copy.append(ind[:])
                fitnesses_copy.append(ind.fitness.values)

            gen_info = {
                'generation': gen,
                'population': population_copy,  # Store offspring individuals
                'fitnesses': fitnesses_copy  # Store fitness values of offspring
            }

            gen_history.append(gen_info)

            # Compile and record current generation statistics
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

        # Return the final population, logbook, and generation history
        return population, logbook, gen_history

    def save_generation_history(self, gen_history):
        gen_history_dict = []
        for gen_info in gen_history:
            feasible_population = []
            feasible_fitnesses = []
            for ind, fitness in zip(gen_info['population'], gen_info['fitnesses']):
                if fitness[0] != -1e9 and fitness[1] != -1e9:
                    feasible_population.append(ind)
                    feasible_fitnesses.append(fitness)
            
            population_strings = []
            for ind in feasible_population:
                ind_str_elements = []
                for x in ind:
                    ind_str_elements.append(str(x))
                ind_str = " ".join(ind_str_elements)
                population_strings.append(ind_str)

            fitnesses_list = []
            for fitness in feasible_fitnesses:
                fitness_list = list(fitness)
                fitnesses_list.append(fitness_list)

            gen_history_dict.append({
                'generation': gen_info['generation'],
                'population': population_strings,
                'fitnesses': fitnesses_list
            })

        with open('crossover2_graph_figures_jsons/generation_history.json', 'w') as f:
            json.dump(gen_history_dict, f, indent=4)

    def save_crossover_info(self):
        # Write the crossover information to a JSON file
        crossover_info = self.crossover_info
        output_data = []

        for idx, info in enumerate(crossover_info):
            ind1 = info['parent1']
            ind2 = info['parent2']
            
            if isinstance(ind1, Iterable) and isinstance(ind2, Iterable):
                before_changes_parent1 = [" ".join(map(str, ind1))]
                before_changes_parent2 = [" ".join(map(str, ind2))]
            else:
                before_changes_parent1 = [ind1]
                before_changes_parent2 = [ind2]

            ind1 = info['offspring1']
            ind2 = info['offspring2']
            
            if isinstance(ind1, Iterable) and isinstance(ind2, Iterable):
                after_changes_parent1 = [" ".join(map(str, ind1))]
                after_changes_parent2 = [" ".join(map(str, ind2))]
            else:
                after_changes_parent1 = [ind1]
                after_changes_parent2 = [ind2]

            sorted_subgraph_nodes = sorted(info['subgraph_nodes'])
            json_data = {
                "Crossover": idx + 1,
                "Parent 1": before_changes_parent1,
                "Parent 2": before_changes_parent2,
                "Offspring 1": after_changes_parent1,
                "Offspring 2": after_changes_parent2,
                "Subgraph Nodes": sorted_subgraph_nodes,
                "Selected Random Node": info['selected_node'],
                "Selected Subgraph Parent": info['selected_subgraph_parent']
            }
            output_data.append(json_data)

        with open('crossover2_graph_figures_jsons/crossover_info.json', 'w') as outfile:
            json.dump(output_data, outfile, indent=4)

    def visualize_pareto_front(self, gen_history, best_solution):
        last_generation = gen_history[-1]
        feasible_population = []
        feasible_fitnesses = []
        best_solution_fitness = None
        for ind, fitness in zip(last_generation['population'], last_generation['fitnesses']):
            if fitness[0] != -1e9 and fitness[1] != -1e9:
                ind_with_fitness = creator.Individual(ind)  # Create an Individual object
                ind_with_fitness.fitness.values = fitness  # Set the fitness attribute
                feasible_population.append(ind_with_fitness)  # Append the Individual to the feasible_population
                feasible_fitnesses.append(fitness)
                if ind == best_solution:
                    best_solution_fitness = fitness

        # Calculate Pareto front
        pareto_front = tools.sortNondominated(feasible_population, len(feasible_population))[0]
        pareto_front_fitnesses = [ind.fitness.values for ind in pareto_front]

        # Separate dominated and non-dominated solutions
        dominated = [fit for fit in feasible_fitnesses if fit not in pareto_front_fitnesses]
        non_dominated = [fit for fit in feasible_fitnesses if fit in pareto_front_fitnesses]

        plt.figure()
        plt.scatter(*zip(*dominated), marker='o', s=30, edgecolor='k', color='gray', label='Dominated solutions')
        plt.scatter(*zip(*non_dominated), marker='o', s=30, edgecolor='k', color='blue', label='Non-dominated solutions')
        
        if best_solution_fitness is not None:
            plt.scatter(best_solution_fitness[0], best_solution_fitness[1], marker='o', s=30, color='red', edgecolor='k', label='Best solution')

        plt.xlabel("Total Utility")
        plt.ylabel("Total Cost")
        plt.title("Pareto Front")
        plt.legend()
        plt.savefig('crossover2_graph_figures_jsons/pareto_front.png', bbox_inches='tight', dpi=300)
        plt.show()

    def run_optimization(self):
        # Set the total number of charging stations that can be installed
        total_charging_stations = round(self.graph_generator.num_nodes * 10)
        print(f"Maximum number of charging stations that can be installed: {total_charging_stations}")

        # Set the minimum and maximum number of charging stations based on the total number of charging stations
        min_charging_stations = 0
        max_charging_stations = round((total_charging_stations / self.graph_generator.num_nodes) * 2)
        
        # Run NSGA-II
        best_solution, gen_history = self.optimize_charging_stations(self.graph_generator.utility_cost_data, self.graph_generator.num_nodes, total_charging_stations, min_charging_stations, max_charging_stations)

        # Write the crossover information to a JSON file
        self.save_crossover_info()

        # Visualize the Pareto front
        self.visualize_pareto_front(gen_history, best_solution)

        # Save the generation history to a file in JSON format
        self.save_generation_history(gen_history)

        # Visualize the best solution found by NSGA-II
        self.visualize_charging_stations(self.graph_generator.utility_cost_data, best_solution)

        print(f"Total number of charging stations that are installed: {sum(best_solution)}")

        # Save the best solution to a file in JSON format
        self.save_best_solution(best_solution)