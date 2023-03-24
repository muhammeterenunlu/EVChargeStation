from graph_generator import GraphGenerator
from nsga2_optimizer import optimize_charging_stations
import json

def main():
    print("Please wait for the process. Graphs are being created...")
    generate_graph = GraphGenerator()
    generate_graph.static_graph_generator()
    generate_graph.dynamic_graph_generator()
    generate_graph.write_static_json()
    generate_graph.write_dynamics_json()
    generate_graph.kmeans_clustering()
    total_charging_stations = round(generate_graph.num_nodes * 10)
    min_charging_stations = 1
    max_charging_stations = max(1, round(total_charging_stations / generate_graph.num_nodes))

    best_solution, best_utility, best_cost = optimize_charging_stations(generate_graph.utility_cost_data, generate_graph.num_nodes, total_charging_stations, min_charging_stations, max_charging_stations)

    # Print the results
    print(f"Best solution: {best_solution}")
    print(f"Best utility: {best_utility}")
    print(f"Best cost: {best_cost}")

    # Optional: Save the best solution to a file
    with open('best_solution.json', 'w') as f:
        json.dump({'Node': list(range(generate_graph.num_nodes)), 'Charging Stations': best_solution}, f, indent=4)
    
if __name__ == "__main__":
    main()