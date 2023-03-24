from graph_generator import GraphGenerator
from nsga2_optimizer import optimize_charging_stations
import json

def main():
    # Generate the initial static graph and the dynamic graphs
    print("Please wait for the process. Graphs are being created...")
    generate_graph = GraphGenerator()
    generate_graph.static_graph_generator()
    generate_graph.dynamic_graph_generator()
    generate_graph.write_static_json()
    generate_graph.write_dynamics_json()
    generate_graph.kmeans_clustering()
    

    # Set the total number of charging stations that can be installed
    total_charging_stations = round(generate_graph.num_nodes * 10)
    print(f"Maximum number of charging stations that can be installed: {total_charging_stations}")
  
    # Set the minimum and maximum number of charging stations based on the total number of charging stations
    min_charging_stations = 1
    max_charging_stations = round((total_charging_stations / generate_graph.num_nodes)*2)

    # Run NSGA-II
    best_solution, best_utility, best_cost = optimize_charging_stations(generate_graph.utility_cost_data, generate_graph.num_nodes, total_charging_stations, min_charging_stations, max_charging_stations)

    print(f"Total number of charging stations that are installed: {sum(best_solution)}")
    
    # Print the results of NSGA-II
    print(f"Best solution: {best_solution}")
    print(f"Best utility: {best_utility}")
    print(f"Best cost: {best_cost}")

    # Save the best solution to a file in JSON format
    with open('best_solution.json', 'w') as f:
        json.dump({'Node': list(range(generate_graph.num_nodes)), 'Charging Stations': best_solution}, f, indent=4)
    
if __name__ == "__main__":
    main()