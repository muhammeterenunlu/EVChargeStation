import random
import json
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans

class GraphGenerator:

    # init or constructor method to initialize the class
    def __init__(self):
        # Generate a random graph with random number of nodes and edges
        self.num_nodes = random.randint(10, 20)
        self.num_edges = random.randint(self.num_nodes, self.num_nodes * (self.num_nodes - 1) // 2)

        # Create the initial static graph
        self.G = nx.gnm_random_graph(self.num_nodes, self.num_edges)

        # Define the number of dynamic graphs to create
        self.num_dynamic_graphs = random.randint(500,1000)

        # Create an empty list to store the data of the initial static graph
        self.static_graph_data = []

        # Create an empty list to store the data of the dynamic graphs
        self.dynamic_graphs_data = []

        # Create an empty list to store the data of the clusters
        self.utility_cost_data = []

    def static_graph_generator(self):
        # Add random attributes to each node in the initial static graph
        for node in self.G.nodes:
            population = random.randint(100000, 1000000)
            traffic = random.uniform(0, 1)
            network = random.uniform(0, 1)
            transportation_cost = random.uniform(100, 1000)
            charge_station_cost = random.uniform(1000, 10000)
            self.G.nodes[node]['population'] = population
            self.G.nodes[node]['traffic'] = traffic
            self.G.nodes[node]['network'] = network
            self.G.nodes[node]['transportation cost'] = transportation_cost
            self.G.nodes[node]['charge station cost'] = charge_station_cost
            self.G.nodes[node]['total cost'] = transportation_cost + charge_station_cost
            # Add the initial static graph information to the list
            self.static_graph_data.append({
                 'Graph': 'Initial', 
                 'Node': node, 
                 'Population': population, 
                 'Traffic': traffic, 
                 'Network': network, 
                 'Population Change': 0, 
                 'Traffic Change': 0, 
                 'Network Change': 0, 
                 'Transportation Cost of 1 CS': transportation_cost,
                 'Charge Station Cost of 1 CS': charge_station_cost,
                 'Total Cost of 1 CS': transportation_cost + charge_station_cost})

    def dynamic_graph_generator(self):
        previous_dg = None
        population_change = {}
        traffic_change = {}
        network_change = {}

        # Create dynamic graphs
        for i in range(self.num_dynamic_graphs):
            if i == 0:
                # Create the first dynamic graph based on the initial static graph
                dg = self.G.copy()
            else:
                # Create a new dynamic graph based on the previous dynamic graph
                dg = dg.copy()
            try:
                # Pick a random edge and transfer a random amount of the attributes between the two nodes connected to that edge, while maintaining their sum
                edge = random.choice(list(dg.edges))
                u, v = edge
                transfer_amount = random.uniform(0, 1)
                sum_traffic = dg.nodes[u]['traffic'] + dg.nodes[v]['traffic']
                sum_network = dg.nodes[u]['network'] + dg.nodes[v]['network']
                sum_population = dg.nodes[u]['population'] + dg.nodes[v]['population']
                new_u_traffic = (sum_traffic - transfer_amount * sum_traffic) / 2
                new_v_traffic = sum_traffic - new_u_traffic
                new_u_network = (sum_network - transfer_amount * sum_network) / 2
                new_v_network = sum_network - new_u_network
                new_u_population = (sum_population - transfer_amount * sum_population) / 2
                new_v_population = sum_population - new_u_population
                dg.nodes[u]['traffic'] = new_u_traffic
                dg.nodes[v]['traffic'] = new_v_traffic
                dg.nodes[u]['network'] = new_u_network
                dg.nodes[v]['network'] = new_v_network
                dg.nodes[u]['population'] = new_u_population
                dg.nodes[v]['population'] = new_v_population
            except KeyError:
                print("Error: KeyError occurred while trying to update node attributes.")
                continue

            # Calculate population, traffic and network changes for each node
            if previous_dg is None:
                for node in dg.nodes:
                    population_change[node] = dg.nodes[node]['population'] - self.G.nodes[node]['population']
                    traffic_change[node] = dg.nodes[node]['traffic'] - self.G.nodes[node]['traffic']
                    network_change[node] = dg.nodes[node]['network'] - self.G.nodes[node]['network']
            elif previous_dg is not None:
                for node in dg.nodes:
                    population_change[node] = dg.nodes[node]['population'] - previous_dg.nodes[node]['population']
                    traffic_change[node] = dg.nodes[node]['traffic'] - previous_dg.nodes[node]['traffic']
                    network_change[node] = dg.nodes[node]['network'] - previous_dg.nodes[node]['network']

            previous_dg = dg.copy()

            # Add the dynamic graph information to the list
            for node in dg.nodes:
                population = dg.nodes[node]['population']
                traffic = dg.nodes[node]['traffic']
                network = dg.nodes[node]['network']
                transportation_cost = dg.nodes[node]['transportation cost']
                charge_station_cost = dg.nodes[node]['charge station cost']
                self.dynamic_graphs_data.append({
                     'Graph': f'Dynamic {i+1}', 
                     'Node': node, 
                     'Population': population, 
                     'Traffic': traffic, 
                     'Network': network, 
                     'Population Change': population_change[node], 
                     'Traffic Change': traffic_change[node], 
                     'Network Change': network_change[node], 
                     'Transportation Cost of 1 CS': transportation_cost,
                     'Charge Station Cost of 1 CS': charge_station_cost,
                     'Total Cost of 1 CS': transportation_cost + charge_station_cost})

        print(f"{self.num_nodes} nodes and {self.num_edges} edges generated for the initial static graph.")
        print(f"{self.num_dynamic_graphs} dynamic graphs generated.")

    # Write the data of the initial static graph to a JSON file
    def write_static_json(self):
        with open('static_graph_data.json', 'w') as f:
            json.dump(self.static_graph_data, f, indent=4)
    
    # Write the data of the dynamic graphs to a JSON file
    def write_dynamics_json(self):
        with open('dynamic_graphs_data.json', 'w') as f:
            json.dump(self.dynamic_graphs_data, f, indent=4)


    # K-means clustering. Each cluster represents utility of a charge station.
    def kmeans_clustering(self):
        averaged_data = []
        n_clusters = random.randint(round(self.num_nodes/4),round(self.num_nodes/2))
        # Load JSON data 
        with open('dynamic_graphs_data.json') as f:
            self.dynamic_graphs_data = json.load(f)

        # Compute average values for each node
        nodes = set(datum['Node'] for datum in self.dynamic_graphs_data)

        for node in nodes:
            population_sum = 0
            traffic_sum = 0
            network_sum = 0
            population_change_sum = 0
            traffic_change_sum = 0
            network_change_sum = 0
            total_cost_sum = 0
            count = 0
            for datum in self.dynamic_graphs_data:
                if datum['Node'] == node:
                    population_sum += datum['Population']
                    traffic_sum += datum['Traffic']
                    network_sum += datum['Network']
                    population_change_sum += datum['Population Change']
                    traffic_change_sum += datum['Traffic Change']
                    network_change_sum += datum['Network Change']
                    total_cost_sum += datum['Total Cost of 1 CS']
                    count += 1
            averaged_data.append({
                'Node': node,
                'Population': population_sum / count,
                'Traffic': traffic_sum / count,
                'Network': network_sum / count,
                'Population_change': population_change_sum / count,
                'Traffic_change': traffic_change_sum / count,
                'Network_change': network_change_sum / count,
                'Total Cost of 1 CS': total_cost_sum / count})

        # Convert data to a 2D numpy array for clustering
        X = np.array([[datum['Population'], datum['Traffic'], datum['Network'], datum['Population_change'], datum['Traffic_change'], datum['Network_change']] for datum in averaged_data])

        # Apply K-Means clustering algorithm
        kmeans = KMeans(n_clusters, random_state=0).fit(X)
        labels = kmeans.labels_.tolist()

        # Add cluster labels and total cost to the data
        for i, datum in enumerate(averaged_data):
            self.utility_cost_data.append({
                'Utility': labels[i] + 1, 
                'Total Cost': datum['Total Cost of 1 CS']})
            
        # Write cluster data to a JSON file
        with open('utility_cost_data.json', 'w') as f:
            json.dump(self.utility_cost_data, f, indent=4)
