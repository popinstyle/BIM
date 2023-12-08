import time
import random
import numpy as np
from collections import Counter
import pandas as pd
import networkx as nx
from igraph import *


def get_RRS(G, p):
    """
    Inputs: G:  Ex2 dataframe of directed edges. Columns: ['source','target']
            p:  Disease propagation probability
    Return: A random reverse reachable set expressed as a list of nodes
    """

    # Step 1. Select random source node
    source = random.choice(np.unique(G['source']))

    # Step 2. Get an instance of g from G by sampling edges
    g = G.copy().loc[np.random.uniform(0, 1, G.shape[0]) < p]

    # Step 3. Construct reverse reachable set of the random source node
    new_nodes, RRS0 = [source], [source]
    while new_nodes:
        # Limit to edges that flow into the source node
        temp = g.loc[g['target'].isin(new_nodes)]

        # Extract the nodes flowing into the source node
        temp = temp['source'].tolist()

        # Add new set of in-neighbors to the RRS
        RRS = list(set(RRS0 + temp))

        # Find what new nodes were added
        new_nodes = list(set(RRS) - set(RRS0))

        # Reset loop variables
        RRS0 = RRS[:]

    return (RRS)

def ris(G, k, p=0.5, mc=1000):
    """
    Inputs: G:  Ex2 dataframe of directed edges. Columns: ['source','target']
            k:  Size of seed set
            p:  Disease propagation probability
            mc: Number of RRSs to generate
    Return: A seed set of nodes as an approximate solution to the IM problem
    """

    # Step 1. Generate the collection of random RRSs
    start_time = time.time()
    R = [get_RRS(G, p) for _ in range(mc)]

    # Step 2. Choose nodes that appear most often (maximum coverage greedy algorithm)
    SEED, timelapse = [], []
    for _ in range(k):
        # Find node that occurs most often in R and add to seed set
        flat_list = [item for sublist in R for item in sublist]
        seed = Counter(flat_list).most_common()[0][0]
        SEED.append(seed)

        # Remove RRSs containing last chosen seed
        R = [rrs for rrs in R if seed not in rrs]

        # Record Time
        timelapse.append(time.time() - start_time)

    return (sorted(SEED), timelapse)


G = Graph.Barabasi(n=100, m=3,directed=True)

# Transform into dataframe of edges
source_nodes = [edge.source for edge in G.es]
target_nodes = [edge.target for edge in G.es]
df = pd.DataFrame({'source': source_nodes,'target': target_nodes})

# Plot graph
G.es["color"], G.vs["color"] = "#B3CDE3", "#FBB4AE"
plot(G,bbox = (280, 280),margin = 11,layout=G.layout("kk"))