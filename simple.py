import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from RIC import runRICmodel
from RLT import runRLTmodel
from random import random
from Main import runICmodel_n, runLTmodel_n

trueP = {('A', 'B'): 0.8747333918899123, ('A', 'E'): 1, ('B', 'D'): 0.9503718809869643, ('B', 'C'): 1, ('D', 'A'): 0.401434128449473, ('D', 'G'): 0.44201541840202013, ('C', 'A'): 1, ('F', 'A'): 0.353795090619972, ('F', 'G'): 0.39934807659240457, ('H', 'D'): 1, ('H', 'G'): 1}

G = nx.DiGraph()
G.add_edges_from([('A', 'B'), ('A', 'E'), ('B', 'D'), ('B', 'C'), ('C', 'A'), ('D', 'A'), ('D', 'G'), ('F', 'A'), ('F', 'G'), ('H', 'D'), ('H', 'G')])
threshold = {}
for u in list(G.nodes()):
    threshold[u] = random()
G2 = nx.DiGraph()
G2.add_edges_from([('B', 'D'), ('B', 'C'), ('D', 'G'), ('F', 'G'), ('H', 'D'), ('H', 'G')])
pos = nx.shell_layout(G)

subG = nx.DiGraph()
subG.add_edges_from([('A', 'B'), ('A', 'E'), ('B', 'D'), ('B', 'C'), ('C', 'A'), ('D', 'A'), ('D', 'G')])
subG2 = nx.DiGraph()
subG2.add_edges_from([('B', 'D'), ('B', 'C'), ('D', 'G')])

# nx.draw(G, pos, edge_color="grey", node_size=500, with_labels=True)
# plt.show()

ori1 = len(runRLTmodel(G, trueP, threshold))
ori2 = len(runRLTmodel(G2, trueP, threshold))
sub1 = len(runRLTmodel(subG, trueP, threshold))
sub2 = len(runRLTmodel(subG2, trueP, threshold))

print(ori1, ori2, sub1, sub2, (ori1 - ori2) <= (sub1 - sub2))