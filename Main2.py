import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from random import random
from tqdm import tqdm

activation_prob = 0.01

G = nx.read_edgelist('./datasets/rt_occupy/rt_occupy.edges', nodetype=int, create_using=nx.DiGraph())

# undirectedG = nx.read_edgelist('./datasets/dblp/index.txt', nodetype=int, create_using=nx.Graph())


time = 0

# 问题输入：1个图和一个影响范围
# 如：影响到80%的节点种子集的范围在多少

# 随机选择`coverage%`的节点并生成子图
def generateSubGraph(G, coverage):
    nodes = np.random.choice(G.nodes(), int(len(G.nodes()) * coverage), replace=False)
    subG = nx.DiGraph()
    for u in nodes:
        for (u, v) in G.edges(u):
            if v in nodes:
                subG.add_edge(u, v, weight=activation_prob)

    return subG

subG = generateSubGraph(G, 0.01)



# for u in subG.nodes():
#     for (v, u) in subG.in_edges(u):
#         print(v, u)
#
#     print(u, ' u')
#
#     nx.spring_layout(subG)
#     nx.draw(subG, with_labels=True, font_size=16)
#     plt.show()
#
# print(123)


# 问题转化为子图100%覆盖
# 若要得到一个范围，比如30-40个节点，可以采用蒙特卡洛模拟来多次模拟
# def runBICmodel(G):
#     spread = list(G.nodes())
#
#     S = []
#
#     i = 0
#     while i < len(spread):
#         for (v, spread[i]) in G.in_edges(spread[i]):
#             print(v, spread[i])
#             if
#
#     return S

# runBICmodel(subG)

# nx.spring_layout(subG)
# nx.draw(subG, with_labels=True, font_size=10)
# plt.show()


# S = [subG.subgraph(c).copy() for c in nx.weakly_connected_components(subG)]


# for c in S:
#     print(c)
#     # nx.spring_layout(c)
#     # nx.draw(c, with_labels=True, font_size=16)
#     # plt.show()
#
# print(len(S))


for c in sorted(nx.weakly_connected_components(G), key=len, reverse=True):
    print(len(c))


# 社群检测算法，条件受限的BIM算法
# def edge_to_remove(graph):
#     G_dict = nx.edge_betweenness_centrality(graph)
#     edge = ()
#
#     # extract the edge with highest edge betweenness centrality score
#     for key, value in sorted(G_dict.items(), key=lambda item: item[1], reverse = True):
#         edge = key
#         break
#
#     return edge
#
# def girvan_newman(graph):
#     # find number of connected components
#     sg = nx.connected_components(graph)
#     sg_count = nx.number_connected_components(graph)
#
#     while(sg_count == 1):
#         graph.remove_edge(edge_to_remove(graph)[0], edge_to_remove(graph)[1])
#         sg = nx.connected_components(graph)
#         sg_count = nx.number_connected_components(graph)
#
#     return sg
#
#0
# c = girvan_newman(undirectedG.copy())
# node_groups = []
#
# for i in c:
#     node_groups.append(len(l2 ist(i)))
#
#
# print(node_groups)