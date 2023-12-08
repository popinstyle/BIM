# from diffusion import LinearThreshold, IndependentCascade
import networkx as nx
import pickle

def greedy(graph, diffuse, k):
	S = set()
	A = set(graph.nodes)
	while len(S) < k:
		node_diffusion = {}
		for node in A:
			S.add(node)
			node_diffusion[node] = diffuse.diffuse_mc(S)
			S.remove(node)
		max_node = max(node_diffusion.items(), key=lambda x: x[1])[0]
		S.add(max_node)
		A.remove(max_node)
	return S


# G = pickle.load(open('../datasets/dolphins/Small_Final_SubG.G', 'rb'), encoding='latin1')
#
# IC = IndependentCascade(G)
# print(len(greedy(G, IC, 30)))

