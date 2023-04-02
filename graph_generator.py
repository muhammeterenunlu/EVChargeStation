import random
import json
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans

class GraphGenerator:

    # init or constructor method to initialize the class
    def __init__(self):

        # Generate a random graph with random number of nodes and edges
        self.num_nodes = random.randint(50, 200)
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
            transportation_cost = random.randint(100, 1000)
            charge_station_cost = random.randint(1000, 10000)
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
                
                attributes = ['traffic', 'network', 'population']
                
                # Randomly choose one of the two nodes
                chosen_node, other_node = random.choice([(u, v), (v, u)])
                
                for attr in attributes:
                    sum_attr = dg.nodes[chosen_node][attr] + dg.nodes[other_node][attr]
                    new_chosen_node_attr = int((1 - transfer_amount) * sum_attr)
                    new_other_node_attr = sum_attr - new_chosen_node_attr

                    dg.nodes[chosen_node][attr] = new_chosen_node_attr
                    dg.nodes[other_node][attr] = new_other_node_attr

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

        # Compute average values for each node and add them to the averaged_data list
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
                'Total Utility of 1 CS': int(0.3 * population_sum / count + 0.6 * traffic_sum / count + 0.1 * network_sum / count),
                'Total Cost of 1 CS': int(total_cost_sum / count)})

        # Convert data to a 2D numpy array for clustering
        X = np.array([
            [
                datum['Total Utility of 1 CS'],
                datum['Total Cost of 1 CS']
            ]
            for datum in averaged_data
        ])

        # Apply K-Means clustering algorithm for utility
        kmeans_utility = KMeans(n_clusters, random_state=0, n_init='auto').fit(X[:, [0]]) # Only consider the first column (utility) 
        utility_labels = kmeans_utility.labels_.tolist()

        # Apply K-Means clustering algorithm for cost
        kmeans_cost = KMeans(n_clusters, random_state=0, n_init='auto').fit(X[:, [1]])  # Only consider the second column (cost)
        cost_labels = kmeans_cost.labels_.tolist()

        # Reverse the utility cluster labels so that higher labels have higher utility
        max_utility_label = max(utility_labels)
        utility_labels = [max_utility_label - label for label in utility_labels]

        # Reverse the cost cluster labels so that higher label represents higher cost
        max_cost_label = max(cost_labels)
        cost_labels = [max_cost_label - label for label in cost_labels]

        # Add utility and cost cluster labels and total cost to the data
        for i, datum in enumerate(averaged_data):
            self.utility_cost_data.append({
                'Location': datum['Node'],
                'Utility Specified By Using ML': utility_labels[i] + 1,
                'Cost Specified By Using ML': cost_labels[i] + 1,
                'Total Utility of 1 CS': datum['Total Utility of 1 CS'],
                'Total Cost of 1 CS': datum['Total Cost of 1 CS']})
            
        # Write cluster data to a JSON file
        with open('utility_cost_data.json', 'w') as f:
            json.dump(self.utility_cost_data, f, indent=4)
