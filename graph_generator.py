import random
import json
import itertools
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines
from sklearn.cluster import KMeans
from kneed import KneeLocator

class GraphGenerator:

    def __init__(self):
        connected = False
        self.edge_creation_prob = 0.05
        self.num_nodes = 50

        while not connected:
            # Generate a random graph
            self.G = nx.gnp_random_graph(self.num_nodes, self.edge_creation_prob)

            # Ensure the minimum degree of each node is at least 2
            for node in self.G.nodes():
                if self.G.degree(node) < 2:
                    connected_neighbors = list(self.G.neighbors(node))
                    available_nodes = set(self.G.nodes()) - set(connected_neighbors) - {node}
                    new_edges = random.sample(available_nodes, 2 - self.G.degree(node))
                    for new_edge in new_edges:
                        self.G.add_edge(node, new_edge)

            # Check if the graph is connected
            connected = nx.is_connected(self.G)

        # Define the number of dynamic graphs to create
        self.num_dynamic_graphs = random.randint(1000,2000)

        # Create an empty list to store the data of the initial static graph
        self.static_graph_data = []

        # Create an empty list to store the data of the dynamic graphs
        self.dynamic_graphs_data = []

        # Create an empty list to store the data of the clusters
        self.utility_cost_data = []

        # Ensure the number of dynamic graphs is divisible by the number of edges
        num_edges = self.G.number_of_edges()
        while self.num_dynamic_graphs % num_edges != 0:
            self.num_dynamic_graphs += 1

    def static_graph_generator(self):
        # Add random attributes to each node in the initial static graph
        for node in self.G.nodes:
            population = random.randint(50000, 100000)
            traffic = random.uniform(0, 1)
            network = random.uniform(0, 1)
            traffic = traffic * population
            network = network * population
            transportation_cost = random.randint(500, 1000)
            charge_station_cost = random.randint(5000, 10000)
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
        # Assign densities to the edges of the static graph
        self.assign_edge_densities()

    # Function to assign density categories and flow ranges to the edges in the static graph
    def assign_edge_densities(self):
        for edge in self.G.edges:
            density_category = random.choice(['VERY DENSE', 'MEDIUM DENSE', 'LOW DENSE'])

            if density_category == 'VERY DENSE':
                flow_range = (0.65, 0.95)
            elif density_category == 'MEDIUM DENSE':
                flow_range = (0.35, 0.65)
            elif density_category == 'LOW DENSE':
                flow_range = (0.05, 0.35)

            lower_bound, upper_bound = flow_range
            self.G.edges[edge]['density_category'] = density_category
            self.G.edges[edge]['flow'] = random.uniform(lower_bound, upper_bound)
    
    def visualize_initial_static_graph(self):
        pos = nx.spring_layout(self.G)

        # Add a title to the plot
        plt.title('Initial Static Graph')

        # Define edge colors based on their densities
        edge_colors = []
        for edge in self.G.edges:
            density = self.G.edges[edge]['density_category']
            if density == 'VERY DENSE':
                edge_colors.append('red')
            elif density == 'MEDIUM DENSE':
                edge_colors.append('orange')
            elif density == 'LOW DENSE':
                edge_colors.append('yellow')

        nx.draw(self.G, pos, with_labels=True, node_color='skyblue', edge_color=edge_colors, node_size=250, font_size=6, font_weight='bold', width=1.5)

        # Add a legend explaining the edge colors
        legend_elements = [matplotlib.lines.Line2D([0], [0], color='red', lw=2, label='VERY DENSE'),
                        matplotlib.lines.Line2D([0], [0], color='orange', lw=2, label='MEDIUM DENSE'),
                        matplotlib.lines.Line2D([0], [0], color='yellow', lw=2, label='LOW DENSE')]

        plt.legend(handles=legend_elements, loc='lower right', fontsize='small', title='Edge Density')

        # Save the visualization to a file
        plt.savefig('initial_static_graph.png', dpi=300)
        plt.show()

    def write_connected_nodes_json(self):
        connected_nodes = {}
        node_ids = sorted(self.G.nodes)

        for node_id in node_ids:
            connected_nodes[f'Node {node_id}'] = []

        for edge in self.G.edges:
            connected_nodes[f'Node {edge[0]}'].append(edge[1])
            connected_nodes[f'Node {edge[1]}'].append(edge[0])

        # Write the connected nodes data to a JSON file
        with open('connected_nodes.json', 'w') as f:
            json.dump(connected_nodes, f, indent=4)

    def dynamic_graph_generator(self):
        previous_dg = None
        population_change = {}
        traffic_change = {}
        network_change = {}
        # Create a dictionary to count how many times each node has been selected
        #self.selected_node_counter = {node: 0 for node in range(self.num_nodes)}
        edge_list = list(self.G.edges)
        # = {edge: 0 for edge in edge_list}  # Initialize a dictionary to store edge selection counts

        # Create dynamic graphs
        for i in range(self.num_dynamic_graphs):
            if i % len(edge_list) == 0:
                random.shuffle(edge_list)  # Shuffle the edge list every time it completes a cycle
                edge_cycle = itertools.cycle(edge_list)
            if i == 0:
                # Create the first dynamic graph based on the initial static graph
                dg = self.G.copy()
            else:
                # Create a new dynamic graph based on the previous dynamic graph
                dg = dg.copy()

            # Get the next edge in the cycle
            edge = next(edge_cycle)
            #edge_selection_count[edge] += 1
            #print(edge)

            try:
                u, v = edge
                density_category = self.G.edges[edge]['density_category']

                # Define the transfer range based on the density category
                if density_category == 'VERY DENSE':
                    transfer_range = (0.65, 0.95)
                elif density_category == 'MEDIUM DENSE':
                    transfer_range = (0.35, 0.65)
                elif density_category == 'LOW DENSE':
                    transfer_range = (0.05, 0.35)
                
                transfer_amount = random.uniform(*transfer_range)
                    
                attributes = ['traffic', 'network', 'population']
                
                # Randomly choose one of the two nodes
                chosen_node, other_node = random.choice([(u, v), (v, u)])

                # Increment the selected node counter
                #self.selected_node_counter[chosen_node] += 1
                #self.selected_node_counter[other_node] += 1
                
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

        print(f"{self.num_nodes} nodes and {self.G.number_of_edges()} edges generated for the initial static graph.")
        print(f"{self.num_dynamic_graphs} dynamic graphs generated.")
        #print("Selected node counts:", self.selected_node_counter)
        #print("Edge selection count:")
        #for edge, count in edge_selection_count.items():
            #print(f"{edge}: {count}")

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
        # Load JSON data 
        with open('dynamic_graphs_data.json') as f:
            self.dynamic_graphs_data = json.load(f)

        # Get the set of nodes
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
                'Total Utility of 1 CS': int(0.25 * (0.3 * population_sum / count + 0.6 * traffic_sum / count + 0.1 * network_sum / count) + 0.75 * (0.3 * abs(population_change_sum) / count + 0.6 * abs(traffic_change_sum) / count + 0.1 * abs(network_change_sum) / count)),
                'Total Cost of 1 CS': int(total_cost_sum / count)})

        # Convert data to a 2D numpy array for clustering
        X = np.array([
            [
                datum['Total Utility of 1 CS'],
                datum['Total Cost of 1 CS']
            ]
            for datum in averaged_data
        ])

        # Determine the optimal number of clusters using the Elbow Method
        sse = []
        k_candidates = range(1, round(self.num_nodes/2) + 1)
        for k in k_candidates:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
            kmeans.fit(X)
            sse.append(kmeans.inertia_)

        # Find the elbow point in the SSE curve
        kl = KneeLocator(k_candidates, sse, curve="convex", direction="decreasing")
        n_clusters = kl.elbow
        print(f"Number of cluster: {n_clusters}")

        # Apply K-Means clustering algorithm for utility (arrange n_init = 10 default value on MAC OS)
        kmeans_utility = KMeans(n_clusters, random_state=0, n_init=10).fit(X[:, [0]]) # Only consider the first column (utility) 
        utility_labels = kmeans_utility.labels_.tolist()

        # Apply K-Means clustering algorithm for cost
        kmeans_cost = KMeans(n_clusters, random_state=0, n_init=10).fit(X[:, [1]])  # Only consider the second column (cost)
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
                'Node': datum['Node'],
                'Utility Specified By Using ML': utility_labels[i] + 1,
                'Cost Specified By Using ML': cost_labels[i] + 1,
                'Total Utility of 1 CS': datum['Total Utility of 1 CS'],
                'Total Cost of 1 CS': datum['Total Cost of 1 CS']})
            
        # Write cluster data to a JSON file
        with open('utility_cost_data.json', 'w') as f:
            json.dump(self.utility_cost_data, f, indent=4)