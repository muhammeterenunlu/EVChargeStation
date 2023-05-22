import random
import numpy as np
import matplotlib.pyplot as plt
import json
from collections.abc import Iterable
from deap import base, creator, tools, algorithms
from deap.tools._hypervolume import hv
from deap.tools.emo import assignCrowdingDist
from population_generator import generate_population

class NSGA2Optimizer2:

    # Add a global variable to store the crossover information
    crossover_info = []
    # Add a variable for population size
    population_size = None

    def __init__(self,graph_generator):
        self.graph_generator = graph_generator
        self.gen_history = []

    # This method sets up and runs the NSGA-II algorithm, defining objective functions, constraints, crossover, mutation, and selection functions. 
    # It returns the best solution found and the generation history.
    def optimize_charging_stations(self, utility_cost_data, num_nodes, total_charging_stations, min_charging_stations, max_charging_stations):
        self.total_charging_stations = total_charging_stations
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

        # Initialize a DEAP toolbox object that will store various components required for the NSGA-II algorithm
        toolbox = base.Toolbox()

        # Define evaluation and selection functions
        def evaluate(individual):
            return total_utility(individual), total_cost(individual)

        # Define a function that checks whether an individual (a solution) satisfies the constraint that the total number of charging stations should not exceed the specified limit
        def feasible(individual):
            self.reduce_excess_charging_stations(individual)
            return True

        # Register 'mate'(crossover) as a function that performs graph crossover on two individuals
        toolbox.register("mate", self.subgraph_crossover)

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
        population_size = 200
        generations = 200
        crossover_probability = 0.9
        mutation_probability = 0.1

        # Save population_size to the class variable
        self.population_size = population_size
        
        # Print the parameters
        print("Population size:", population_size, end="  ")
        print("Generations:", generations, end="  ")
        print("Crossover probability:", crossover_probability, end="  ")
        print("Mutation probability:", mutation_probability)

        pop = generate_population(num_nodes, min_charging_stations, max_charging_stations, population_size)  # Use the same seed

        pop, _, gen_history = self.custom_eaMuPlusLambda(pop, toolbox, mu=population_size, lambda_=population_size, cxpb=crossover_probability, mutpb=mutation_probability, ngen=generations, verbose=False)
        
        self.gen_history = gen_history  # Set self.gen_history here

        return gen_history
    
    def subgraph_crossover(self, ind1, ind2):
        subgraphs_parent1 = [self.graph_generator.generate_connected_subgraph() for _ in range(10)]
        subgraphs_parent2 = [self.graph_generator.generate_connected_subgraph() for _ in range(10)]
        all_subgraphs = subgraphs_parent1 + subgraphs_parent2
        def subgraph_score(subgraph):
            total_utility = 0
            total_cost = 0
            for node in subgraph.nodes:
                total_utility += self.graph_generator.utility_cost_data[node]['Utility Specified By Using ML']
                total_cost += self.graph_generator.utility_cost_data[node]['Cost Specified By Using ML']
            return total_utility / total_cost
        scores = [subgraph_score(subgraph[0]) for subgraph in all_subgraphs]
        total_score = sum(scores)
        prob = [score / total_score for score in scores]
        chosen_subgraph_idx = np.random.choice(range(len(all_subgraphs)), p=prob)
        best_subgraph = all_subgraphs[chosen_subgraph_idx]

        # Perform crossover using the best subgraph
        subgraph_nodes = list(best_subgraph[0].nodes)
        temp1 = ind1[:]
        temp2 = ind2[:]
        for node in subgraph_nodes:
            ind1[node], ind2[node] = ind2[node], ind1[node]

        # Calculate the scores for the remaining nodes
        def node_score(node):
            return self.graph_generator.utility_cost_data[node]['Utility Specified By Using ML'] / self.graph_generator.utility_cost_data[node]['Cost Specified By Using ML']

        remaining_nodes = [node for node in range(len(ind1)) if node not in subgraph_nodes]
        remaining_nodes_scores = [node_score(node) for node in remaining_nodes]
        total_score_remaining = sum(remaining_nodes_scores)
        prob_remaining = [score / total_score_remaining for score in remaining_nodes_scores]

        # Perform uniform crossover on the remaining nodes with probability-based swapping
        for i, node in enumerate(remaining_nodes):
            if np.random.rand() < prob_remaining[i]:  # swapping probability based on the node score
                if np.random.rand() < 0.75:  # overall 50% chance for crossover
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

    def custom_eaMuPlusLambda(self, population, toolbox, mu, lambda_, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals']
        if stats:
            logbook.header.extend(stats.fields)
        invalid_ind = []
        for ind in population:
            if not ind.fitness.valid:
                invalid_ind.append(ind)
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        if halloffame is not None:
            halloffame.update(population)
        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
        gen_history = []
        for gen in range(1, ngen + 1):
            offspring = toolbox.select(population, lambda_)
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
            population = population + offspring  # Combine parents and offspring
            invalid_ind = []
            for ind in population:
                if not ind.fitness.valid:
                    invalid_ind.append(ind)
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            if halloffame is not None:
                halloffame.update(population)
            population = toolbox.select(population, mu)  # Select the best individuals based on dominance
            assignCrowdingDist(population)  # Assign crowding distances
            population_copy = []
            fitnesses_copy = []
            for ind in population:
                population_copy.append(ind[:])
                fitnesses_copy.append(ind.fitness.values)
            gen_info = {
                'generation': gen,
                'population': population_copy,  # Store the selected individuals
                'fitnesses': fitnesses_copy  # Store fitness values of selected individuals
            }
            gen_history.append(gen_info)
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)
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

    def visualize_pareto_front(self, gen_history):
        last_generation = gen_history[-1]
        feasible_population = []
        feasible_fitnesses = []
        for ind, fitness in zip(last_generation['population'], last_generation['fitnesses']):
            if fitness[0] != -1e9 and fitness[1] != -1e9:
                ind_with_fitness = creator.Individual(ind)  # Create an Individual object
                ind_with_fitness.fitness.values = fitness  # Set the fitness attribute
                feasible_population.append(ind_with_fitness)  # Append the Individual to the feasible_population
                feasible_fitnesses.append(fitness)

        # Calculate Pareto front
        pareto_front = tools.sortNondominated(feasible_population, len(feasible_population))[0]
        pareto_front_fitnesses = [ind.fitness.values for ind in pareto_front]

        # Separate dominated and non-dominated solutions
        dominated = [fit for fit in feasible_fitnesses if fit not in pareto_front_fitnesses]
        non_dominated = [fit for fit in feasible_fitnesses if fit in pareto_front_fitnesses]

        plt.figure()

        if dominated:
            plt.scatter(*zip(*dominated), marker='o', s=30, edgecolor='k', color='gray', label='Dominated solutions')
        
        plt.scatter(*zip(*non_dominated), marker='o', s=30, edgecolor='k', color='blue', label='Non-dominated solutions')

        plt.xlabel("Total Utility")
        plt.ylabel("Total Cost")
        plt.title("Pareto Front")
        plt.legend()
        plt.savefig('crossover2_graph_figures_jsons/pareto_front.png', bbox_inches='tight', dpi=300)
        plt.show()

    """
    def calculate_hypervolume(self, front, reference_point):
        # Convert the front and the reference point to NumPy arrays
        front = np.array(front)
        reference_point = np.array(reference_point)
        # Compute and return the hypervolume
        hypervolume = hv.hypervolume(front, reference_point)
        print(f'Hypervolume: {hypervolume}') # Print statement for debugging
        return hypervolume

    def get_overall_reference_point(self, gen_history):
        all_costs = []
        all_utilities = []
        for gen_info in gen_history:
            front = gen_info['fitnesses']
            all_costs.extend([ind[1] for ind in front])
            all_utilities.extend([ind[0] for ind in front])
        max_costs = max(all_costs)
        min_utility = min(all_utilities)
        return [1000, 250]  # Overall reference point across all generations

    def add_hypervolume_to_history(self, gen_history):
        reference_point = self.get_overall_reference_point(gen_history)
        print(f'Reference point: {reference_point}') # Print statement for debugging
        for gen_info in gen_history:
            front = gen_info['fitnesses']
            # Calculate and store the hypervolume using the overall reference point
            self.calculate_hypervolume(front, reference_point)
    """
    # add a new function for reducing excess charging stations based on utility/cost ratio
    def reduce_excess_charging_stations(self, individual):
        while sum(individual) > self.total_charging_stations:
            min_ratio = float('inf')
            min_node = None
            for i in range(self.graph_generator.num_nodes):
                if individual[i] > 0:
                    ratio = self.graph_generator.utility_cost_data[i]['Utility Specified By Using ML'] / self.graph_generator.utility_cost_data[i]['Cost Specified By Using ML']
                    if ratio < min_ratio:
                        min_ratio = ratio
                        min_node = i
            if min_node is not None:
                individual[min_node] -= 1

    def run_optimization(self):
        # Set the total number of charging stations that can be installed
        total_charging_stations = round(self.graph_generator.num_nodes * 10)
        print(f"Maximum number of charging stations that can be installed (Crossover Strategy 2): {total_charging_stations}")
        
        # Set the minimum and maximum number of charging stations based on the total number of charging stations
        min_charging_stations = 0
        max_charging_stations = round((total_charging_stations / self.graph_generator.num_nodes) * 2)
        
        # Run NSGA-II
        gen_history = self.optimize_charging_stations(self.graph_generator.utility_cost_data, self.graph_generator.num_nodes, total_charging_stations, min_charging_stations, max_charging_stations)
        
        # Add hypervolume
        #self.add_hypervolume_to_history(gen_history)

        # Write the crossover information to a JSON file
        self.save_crossover_info()
        
        # Visualize the Pareto front
        self.visualize_pareto_front(gen_history)

        # Save the generation history to a file in JSON format
        self.save_generation_history(gen_history)
