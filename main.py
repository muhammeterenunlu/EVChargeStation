from graph_generator import GraphGenerator
from nsga2_optimizer import NSGA2Optimizer
from nsga2_optimizer2 import NSGA2Optimizer2
import os
#from hypervolume_calculator import add_hypervolume_to_history

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
    #hypervolume1, hypervolume2 = add_hypervolume_to_history(optimize_graph2.gen_history, optimize_graph.gen_history)
    #print("Hypervolume for NSGA2Optimizer2: ", hypervolume1)
    #print("Hypervolume for NSGA2Optimizer: ", hypervolume2)
    
if __name__ == "__main__":
    main()