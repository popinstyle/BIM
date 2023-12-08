import numpy
import networkx as nx
import math
from copy import deepcopy
import pickle
import config


G = pickle.load(open(config.G, 'rb'), encoding='latin1')
communities = []

# G = nx.read_edgelist('./datasets/dolphins/dolphins.mtx', create_using=nx.Graph())

file_object1 = open("./community/dolphins.txt",'r')
try:
  while True:
      line = file_object1.readline().replace('\r','').replace('\n','')
      if line:
          c = line.split(' ')
          toInt = [int(i) for i in c]
          communities.append(toInt)
      else:
          break
finally:
    file_object1.close()

def community_diversity(G, comm):
    CD = {}
    for u in G.nodes():
        value = 0
        neighbors = [v for (u, v) in G.edges(u)]
        for c in comm:
            inC = [i in c for i in neighbors].count(True)
            P_uc = inC / len(neighbors)
            value -= P_uc * math.log(P_uc)

        CD[u] = value

    return CD


def generate_subG(G, nodes):
    subG = deepcopy(G)
    for u in list(subG.nodes()):
        if u not in nodes:
            subG.remove_node(u)
    return subG


def compute_CSR(G, comm, k):
    CSR = {}
    CM = {}
    CD = {}
    # for c in comm:
    #     print(len(c), len(G.nodes()))
    #     CM[c] = len(G.nodes())
    #     CD[c] = len(list(generate_subG(G, c).edges())) / (len(c) * (len(c) - 1) / 2)

    for u in G.nodes():
        com = []
        for (u, v) in G.edges(u):
            for c in comm:
                if v in c:
                    if c not in com:
                        com.append(c)

        val = 1
        neighbors = [v for (u, v) in G.edges(u)]
        for c in com:
            inC = [i in c for i in neighbors].count(True)
            P_uc = inC / len(neighbors)
            x = P_uc * math.log(P_uc)
            y = len(c) / len(list(G.nodes()))
            z = len(list(generate_subG(G, c).edges())) / (len(c) * (len(c) - 1) / 2)
            val += x * y * z

        CSR[u] = len(list(G.edges(u))) * val

    return [u[0] for u in sorted([(u, val) for (u, val) in CSR.items()], key=lambda x:x[1], reverse=True)][:k]


# community_diversity(G, communities)

# CSR = compute_CSR(G, communities, 20)
# print(CSR)

