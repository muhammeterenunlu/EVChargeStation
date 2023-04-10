from graph_generator import GraphGenerator
from nsga2_optimizer import NSGA2Optimizer

import json

def main():
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
    
    # Run optimization and visualization
    optimize_graph = NSGA2Optimizer()
    optimize_graph.run_optimization(generate_graph)
    
if __name__ == "__main__":
    main()