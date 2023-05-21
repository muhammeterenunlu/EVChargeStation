import os
import json
from graph_generator import GraphGenerator
from nsga2_optimizer import NSGA2Optimizer
from nsga2_optimizer2 import NSGA2Optimizer2
from hypervolume_calculator import add_hypervolume_to_history
from inverted_generational_distance import add_igd_to_history, create_reference_set

def main():
    # Create figures folder if not exists
    if not os.path.exists("graph_figures_jsons"):
        os.makedirs("graph_figures_jsons")
    # Create figures2 folder if not exists
    if not os.path.exists("crossover1_graph_figures_jsons"):
        os.makedirs("crossover1_graph_figures_jsons")
    # Create figures3 folder if not exists
    if not os.path.exists("crossover2_graph_figures_jsons"):
        os.makedirs("crossover2_graph_figures_jsons")
    # Create hypervolume and inverted generational distances folder if not exists
    if not os.path.exists("hv&igd_json"):
        os.makedirs("hv&igd_json")
    # Generate the initial static graph and the dynamic graphs
    print("Please wait for the process. Graphs are being created...")
    generate_graph = GraphGenerator()
    generate_graph.static_graph_generator()
    generate_graph.dynamic_graph_generator()
    generate_graph.write_static_json()
    generate_graph.write_connected_nodes_json()
    generate_graph.write_dynamics_json()
    generate_graph.kmeans_clustering()
    generate_graph.visualize_initial_static_graph()
    
    # Run optimization and visualization for crossover 1 (paper)
    optimize_graph = NSGA2Optimizer(generate_graph)
    optimize_graph.run_optimization()

    # Run optimization and visualization for crossover 2 (own)
    optimize_graph2 = NSGA2Optimizer2(generate_graph)
    optimize_graph2.run_optimization()

    # Calculate hypervolumes
    hypervolume1, hypervolume2 = add_hypervolume_to_history(optimize_graph.gen_history, optimize_graph2.gen_history)

    # Save hypervolumes as json
    hypervolumes = {
        "NSGA2Optimizer": hypervolume1,
        "NSGA2Optimizer2": hypervolume2
    }

    with open("hv&igd_json/hypervolumes.json", 'w') as f:
            json.dump(hypervolumes, f, indent=4)

    print("\nHypervolume Results:")
    print("Last generation hypervolume for crossover strategy 1: ", hypervolume1[-1])
    print("Last generation hypervolume for crossover strategy 2: ", hypervolume2[-1])

    # print which crossover is better
    if hypervolume1[-1] > hypervolume2[-1]:
        print("Crossover 1 (paper) is better than crossover 2 according to hypervolume")
    elif hypervolume1[-1] < hypervolume2[-1]:
        print("Crossover 2 (own) is better than crossover 1 according to hypervolume")

    # Calculate IGD
    pareto_optimal_set = create_reference_set([optimize_graph.gen_history, optimize_graph2.gen_history])

    igd1, igd2 = add_igd_to_history(optimize_graph.gen_history, optimize_graph2.gen_history, pareto_optimal_set)

    # Save inverted_generational_distances as json
    inverted_generational_distances = {
        "NSGA2Optimizer": igd1,
        "NSGA2Optimizer2": igd2
    }

    with open("hv&igd_json/inverted_generational_distances.json", 'w') as f:
            json.dump(inverted_generational_distances, f, indent=4)

    print("\nInverted Generational Distance Results:")
    print("Last generation inverted generational distance for crossover strategy 1: ", igd1[-1])
    print("Last generation inverted generational distance for crossover strategy 2: ", igd2[-1])

    # print which crossover is better
    if igd1[-1] < igd2[-1]:
        print("Crossover 1 (paper) is better than crossover 2 according to inverted generational distance")
    elif igd1[-1] > igd2[-1]:
        print("Crossover 2 (own) is better than crossover 1 according to inverted generational distance")
    
if __name__ == "__main__":
    main()