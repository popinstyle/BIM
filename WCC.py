import networkx as nx
from collections import defaultdict

class DirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []

    def addNeighbor(self, nodes):
        self.neighbors += nodes

class Solution:
    """
    @param: nodes: a array of Directed graph node
    @return: a connected set of a directed graph
    """
    def connectedSet2(self, nodes):
        um = {}
        def Find(x):#查找函数
            if x not in um:
                um[x] = x
            father = um[x]
            if father == x:
                return x
            father = Find(father)
            um[x] = father
            return father
        def union(a, b):  #  合并函数
            roota, rootb = Find(a), Find(b)
            if roota != rootb:
                um[min(roota,rootb)]=max(roota,rootb)

        for node in nodes:#将所有连通点用并查集建立图
            Find(node.label)
            for neighbor in node.neighbors:
                union(node.label, neighbor.label)
        k_v = defaultdict(set)
        for roots in um:#查找每个节点所属的块
            father = Find(roots)
            k_v[father].add(roots)
        ans = []
        for root in k_v:
            ans.append(sorted(k_v[root]))
        return ans


nodes = []
file_object1 = open("./datasets/p2p.txt", 'r')
G = nx.DiGraph()
try:
  while True:
      line = file_object1.readline().replace('\n','')
      if line:
          c = line.split('	')
          G.add_edge(c[0], c[1], weight=0.01)
      else:
          break
finally:
    file_object1.close()

nodes = []
for u in G.nodes():
    print(G.out_edges(u))
    nodes.append(DirectedGraphNode(u))




s = Solution()
print(s.connectedSet2(nodes))