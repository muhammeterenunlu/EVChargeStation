from graph_generator import GraphGenerator

def main():
    print("Please wait for the process. Graphs are being created...")
    generate_graph = GraphGenerator()
    generate_graph.static_graph_generator()
    generate_graph.dynamic_graph_generator()
    generate_graph.write_static_json()
    generate_graph.write_dynamics_json()
    generate_graph.kmeans_clustering()
    
if __name__ == "__main__":
    main()