import random
import json
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans

print("Please wait for the process. Graphs are being created...")

# Generate a random graph with random number of nodes and edges
num_nodes = random.randint(10, 20)
num_edges = random.randint(num_nodes, num_nodes * (num_nodes - 1) // 2)
G = nx.gnm_random_graph(num_nodes, num_edges)

# Add random attributes to each node
for node in G.nodes:
    population = random.randint(100000, 1000000)
    traffic = random.uniform(0, 1)
    network = random.uniform(0, 1)
    cost = random.uniform(1000, 10000)
    G.nodes[node]['population'] = population
    G.nodes[node]['traffic'] = traffic
    G.nodes[node]['network'] = network
    G.nodes[node]['cost'] = cost

# Define the number of dynamic graphs to create
num_dynamic_graphs = 1000

# Create an empty list to store the data
data = []

# Add the initial static graph information to the list
for node in G.nodes:
    population = G.nodes[node]['population']
    traffic = G.nodes[node]['traffic']
    network = G.nodes[node]['network']
    cost = G.nodes[node]['cost']
    data.append({'Graph': 'Initial', 'Node': node, 'Population': population, 'Traffic': traffic, 'Network': network, 'Population Change': 0, 'Traffic Change': 0, 'Network Change': 0, 'Cost': cost})

previous_dg = None

# Create dynamic graphs
for i in range(num_dynamic_graphs):
    if i == 0:
        # Create the first dynamic graph based on the initial static graph
        dg = G.copy()
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
    population_change = {}
    traffic_change = {}
    network_change = {}
    if previous_dg is None:
        for node in dg.nodes:
            population_change[node] = dg.nodes[node]['population'] - G.nodes[node]['population']
            traffic_change[node] = dg.nodes[node]['traffic'] - G.nodes[node]['traffic']
            network_change[node] = dg.nodes[node]['network'] - G.nodes[node]['network']
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
        cost = dg.nodes[node]['cost']
        data.append({'Graph': f'Dynamic {i+1}', 'Node': node, 'Population': population, 'Traffic': traffic, 'Network': network, 'Population Change': population_change[node], 'Traffic Change': traffic_change[node], 'Network Change': network_change[node], 'Cost': cost})

print(f"{num_dynamic_graphs} dynamic graphs generated.")

# Write the data to a JSON file
with open('data.json', 'w') as f:
    json.dump(data, f, indent=4)

# k-means clustering

# Load JSON data 
with open('data.json') as f:
    data = json.load(f)

# Compute average values for each node
nodes = set(datum['Node'] for datum in data)

averaged_data = []
for node in nodes:
    population_sum = 0
    traffic_sum = 0
    network_sum = 0
    count = 0
    for datum in data:
        if datum['Node'] == node:
            population_sum += datum['Population']
            traffic_sum += datum['Traffic']
            network_sum += datum['Network']
            count += 1
    averaged_data.append({
        'Node': node,
        'Population': population_sum / count,
        'Traffic': traffic_sum / count,
        'Network': network_sum / count
    })

# Convert data to a 2D numpy array for clustering
X = np.array([[datum['Population'], datum['Traffic'], datum['Network']] for datum in averaged_data])

n_clusters = random.randint(round(num_nodes/4),round(num_nodes/2))

# Apply K-Means clustering algorithm
kmeans = KMeans(n_clusters, random_state=0).fit(X)
labels = kmeans.labels_

# Print results
print(f"Number of clusters: {n_clusters}")
for i, datum in enumerate(averaged_data):
    print(f"Node {datum['Node']}: Cluster {labels[i] + 1}")