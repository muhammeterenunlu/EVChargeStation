import os
import json
from graph_generator import GraphGenerator
from nsga2_optimizer import NSGA2Optimizer
from nsga2_optimizer2 import NSGA2Optimizer2
from hypervolume_calculator import add_hypervolume_to_history

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
    # Create hypervolume folder if not exists
    if not os.path.exists("hypervolumes_json"):
        os.makedirs("hypervolumes_json")
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
    hypervolume1, hypervolume2 = add_hypervolume_to_history(optimize_graph2.gen_history, optimize_graph.gen_history)

    # Save hypervolumes as json
    hypervolumes = {
        "NSGA2Optimizer": hypervolume1,
        "NSGA2Optimizer2": hypervolume2
    }

    with open("hypervolumes_json/hypervolumes.json", 'w') as f:
            json.dump(hypervolumes, f, indent=4)

    print("Last generation hypervolume for NSGA2Optimizer: ", hypervolume1[-1])
    print("Last generation hypervolume for NSGA2Optimizer2: ", hypervolume2[-1])
    
if __name__ == "__main__":
    main()