import networkx as nx
from weight import *
from tqdm import tqdm

def LA(G):
	clusters = []
	vertex = orderVertex(G)
	# Iterate through each vertex
	for v in tqdm(vertex):
		add = False
		# Iterate through all existing cluster to search if current vertex belongs to one of them
		for j in range(len(clusters)):
			U = clusters[j] + [v]
			UW = float(2 * nx.number_of_edges(G.subgraph(U)) / nx.number_of_nodes(G.subgraph(U)))
			W = float(2 * nx.number_of_edges(G.subgraph(clusters[j])) / nx.number_of_nodes(G.subgraph(clusters[j])))
			if UW > W:
				clusters[j] += [v]
				add = True
		# If the vertex doesn't belong to one of the existing cluster, create a new cluster
		if add == False:
			clusters.append([v])
	# Return a list of cluster 
	return clusters

