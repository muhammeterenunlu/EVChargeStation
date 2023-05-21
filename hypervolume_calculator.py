from deap.tools._hypervolume import hv
import numpy as np

def get_overall_reference_point(gen_histories, buffer=2):
    all_costs = []
    all_utilities = []
    for gen_history in gen_histories:
        for gen_info in gen_history:
            front = gen_info['fitnesses']
            all_costs.extend([ind[1] for ind in front])
            all_utilities.extend([ind[0] for ind in front])
    max_costs = max(all_costs) * buffer
    min_utility = min(all_utilities) * buffer
    return [max_costs, min_utility]

def calculate_hypervolume(front, reference_point):
    front = np.array(front)
    reference_point = np.array(reference_point)
    hypervolume = hv.hypervolume(front, reference_point)
    return hypervolume

def add_hypervolume_to_history(gen_history1, gen_history2, buffer=2):
    reference_point = get_overall_reference_point([gen_history1, gen_history2], buffer)
    print("Reference point: ", reference_point)
    hypervolume1 = []
    hypervolume2 = []
    for gen_info in gen_history1:
        front = gen_info['fitnesses']
        x = calculate_hypervolume(front, reference_point)
        hypervolume1.append(x)
    for gen_info in gen_history2:
        front = gen_info['fitnesses']
        y = calculate_hypervolume(front, reference_point)
        hypervolume2.append(y)
    return hypervolume1, hypervolume2
