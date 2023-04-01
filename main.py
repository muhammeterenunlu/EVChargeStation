from graph_generator import GraphGenerator
from nsga2_optimizer import optimize_charging_stations, visualize_charging_stations

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
    best_solution = optimize_charging_stations(generate_graph.utility_cost_data, generate_graph.num_nodes, total_charging_stations, min_charging_stations, max_charging_stations)
    
    # Visualize the best solution found by NSGA-II
    visualize_charging_stations(generate_graph.utility_cost_data, best_solution)

    print(f"Total number of charging stations that are installed: {sum(best_solution)}")

    # Save the best solution to a file in JSON format
    best_solution_dict = {
        'Total Number of Installed Charging Stations': sum(best_solution),
        'Total Utility': sum(generate_graph.utility_cost_data[i]['Total Utility of 1 CS'] * best_solution[i] for i in range(generate_graph.num_nodes)),
        'Total Cost': int(sum(generate_graph.utility_cost_data[i]['Total Cost of 1 CS'] * best_solution[i] for i in range(generate_graph.num_nodes))),
        'Charging Stations': {f'Node {i}': best_solution[i] for i in range(generate_graph.num_nodes)}
    }
    with open('best_solution.json', 'w') as f:
        json.dump(best_solution_dict, f, indent=4)
    
if __name__ == "__main__":
    main()