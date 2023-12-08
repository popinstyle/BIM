from copy import deepcopy
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def runRLTmodel(G, trueP, threshold):
    T = sorted(list(G.nodes()), key=lambda x: G.in_degree(x), reverse=True)
    temp_G = deepcopy(G)
    current_threshold = {}
    total = {}
    remove_node = []
    remove_edge = []
    for (u, v) in list(G.in_edges()):
        if v in total:
            total[v] += trueP[(u, v)]
        else:
            total[v] = trueP[(u, v)]

    for u in list(G.nodes()):
        try:
            current_threshold[u] = str(round(threshold[u], 4)) + '  <  ' + str(round(total[u], 4))
        except KeyError as e:
            current_threshold[u] = str(round(threshold[u], 4))

    # 追溯一下其他节点
    S = []
    for u in T:
        # 节点阈值
        thres = threshold[u]
        thres_total = 0
        for (v, u) in G.in_edges(u):
            thres_total += trueP[(v, u)]
        if thres_total >= thres:
            S.append(u)
            remove_node.append(u)
            # 这是个坑
            # G.remove_node(u)
            # pos = nx.shell_layout(temp_G)
            # nx.draw(G, pos, edge_color="grey", node_size=500, with_labels=True)
            # pos_higher = {}
            # for k, v in pos.items():  # 调整下顶点属性显示的位置，不要跟顶点的序号重复了
            #     if v[1] > 0:
            #         pos_higher[k] = (v[0] - 0.1, v[1] + 0.1)
            #     else:
            #         pos_higher[k] = (v[0] - 0.1, v[1] - 0.1)
            # nx.draw_networkx(temp_G, pos, node_color='r', with_labels=True)
            # nx.draw_networkx_nodes(temp_G, pos, nodelist=remove_node, node_color='c')
            # nx.draw_networkx_labels(temp_G, pos_higher, labels=current_threshold, font_color="brown", font_size=12)
            # nx.draw_networkx_edges(temp_G, pos, edgelist=remove_edge, edge_color="c", arrows=True)
            # plt.show()

    filter_T = []
    for u in list(G.nodes()):
        if u not in S:
            filter_T.append(u)
    return filter_T