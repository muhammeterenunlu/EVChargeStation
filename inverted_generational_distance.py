import numpy as np
from deap import base, creator, tools

# Define your own Fitness class
class Fitness(base.Fitness):
    weights=(1.0, -1.0)

def create_reference_set(gen_histories):
    """Creates a reference set (Pareto optimal set) from given generation histories."""
    all_solutions = []
    for gen_history in gen_histories:
        for gen_info in gen_history:
            front = gen_info['fitnesses']
            # Convert each fitness tuple to an individual with a fitness attribute
            for fit in front:
                ind = creator.Individual(fit)
                ind.fitness = Fitness(values=fit)
                all_solutions.append(ind)
    pareto_optimal_set = tools.sortNondominated(all_solutions, len(all_solutions), first_front_only=True)
    return [list(ind.fitness.values) for ind in pareto_optimal_set[0]]

def calculate_igd(front, pareto_optimal_set):
    """Calculate the Inverted Generational Distance."""
    distances = []
    for optimal_solution in pareto_optimal_set:
        distances_to_solution = np.linalg.norm(front - optimal_solution, axis=1)
        distances.append(np.min(distances_to_solution))
    return np.mean(distances)

def add_igd_to_history(gen_history2, gen_history1, pareto_optimal_set):
    """Calculate IGD for each generation and add to history."""
    igd2 = []
    igd1 = []
    for gen_info in gen_history2:
        front = np.array(gen_info['fitnesses'])
        x = calculate_igd(front, pareto_optimal_set)
        igd2.append(x)
    for gen_info in gen_history1:
        front = np.array(gen_info['fitnesses'])
        y = calculate_igd(front, pareto_optimal_set)
        igd1.append(y)
    return igd2, igd1 