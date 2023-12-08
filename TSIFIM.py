import networkx as nx
import numpy as np
import pickle
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import math

# G = nx.read_edgelist('./datasets/dolphins/dolphins.mtx', create_using=nx.Graph())
activation_probability = 0.1


def initial_candidate_selection(G, k, c):
    candidate = []

    # Construct communicability network matrix


    # Calculate global communicability of each node according to Eq.(4)
    global_comm = {}
    for u in G.nodes():
        prob = 0
        # for (u, v) in G.edges(u):
        #     prob += activation_probability
        for (u, v) in G.edges(u):
            prob += c[u][v]
        global_comm[u] = prob

    leading_nodes = []
    observing_nodes = []
    others = []
    for u in G.nodes():
        leading_status = True
        observing_status = [False, False]
        for (u, v) in G.edges(u):
            if global_comm[u] < global_comm[v]:
                leading_status = False
                observing_status[0] = True
            if global_comm[u] > global_comm[v]:
                observing_status[1] = True
        if all(observing_status):
            observing_nodes.append(u)
        if leading_status:
            leading_nodes.append(u)

    cover_nodes = []
    for u in leading_nodes:
        for (u, v) in G.edges(u):
            cover_nodes.append(v)

    others = [i for i in G.nodes() if i not in cover_nodes]

    # Rank SP in descending order
    SP = sorted([(key, val) for key, val in global_comm.items()], key=lambda x:x[1], reverse=True)

    new_leading_nodes = {}
    for i in SP:
        if i[0] in leading_nodes:
            new_leading_nodes[i[0]] = i[1]


    L = [i for i, val in new_leading_nodes.items()]

    if len(leading_nodes) >= k:
        candidate = L[:k]
    else:
        remains = k - len(L)
        candidate = L + observing_nodes[:remains]

    return candidate, SP


def cal_LRAS(G, c, x, y, x_max, x_min):

    T_1 = (c[x][y] - x_min) / (x_max - x_min)
    T_2 = 0

    nearest_x = [i for (x, i) in G.edges(x)]
    nearest_y = [j for (x, j) in G.edges(y)]
    # z_list = list(set([i for (x, i) in G.edges(x)] + [j for (x, j) in G.edges(y)]))
    z_list = list(set(nearest_x) & set(nearest_y))
    # i->z->j
    N_z = 0
    for z in z_list:
        for (x, t) in G.edges(x):
            if t == z:
                for (z, w) in G.edges(z):
                    if w == y:
                        N_z += 1

    for z in z_list:
        r_iz = (c[x][z] - x_min) / (x_max - x_min)
        r_zj = (c[z][y] - x_min) / (x_max - x_min)
        T_2 += (r_iz * r_zj) / N_z

    T_3 = 0
    # i->x->y->j
    N_xy = 1
    for i in nearest_x:
        for j in nearest_y:
            for (x, t) in G.edges(x):
                if t == i:
                    for (i, w) in G.edges(i):
                        if w == j:
                            for (j, h) in G.edges(j):
                                if h == y:
                                    N_xy += 1

    for i in nearest_x:
        for j in nearest_y:
            r_ix = (c[x][i] - x_min) / (x_max - x_min)
            r_xy = (c[i][j] - x_min) / (x_max - x_min)
            r_yj = (c[j][y] - x_min) / (x_max - x_min)
            T_3 += (r_ix * r_xy * r_yj) / N_xy

    return T_1 + T_2 + T_3


def generate_subG(G, nodes):
    subG = deepcopy(G)
    for u in list(subG.nodes()):
        if u not in nodes:
            subG.remove_node(u)
    # pos = nx.shell_layout(subG)
    # nx.draw_networkx(subG, pos, node_color='r', with_labels=True)
    # plt.show()
    c = nx.communicability(subG)
    return c


def transforms(a, b):
    if a - b > 0:
        return 0
    else:
        return 1


def optimization_candidate(G, c, candidate, k, x_max, x_min):
    cs = []

    i = 1
    labels = {}
    for u in candidate:
        labels[u] = i
        i += 1

    # Calculate the similarity matrix LT
    LT = np.zeros((len(G.nodes()) - k, k))
    non_candidate = [i for i in G.nodes() if i not in candidate]
    i = 0
    for u in non_candidate:
        j = 0
        for v in candidate:
            LT[i][j] = cal_LRAS(G, c, u, v, x_max, x_min)
            j += 1
        i += 1

    # 3 - 10
    for u in G.nodes():
        if u not in candidate:
            LT_max_arr = []
            # 有问题
            line = LT[non_candidate.index(u), :]
            LT_max_val = np.max(line)

            # 判断是否有多个值
            LT_max_arr = [candidate[i] for i, val in enumerate(line) if val == LT_max_val]

            # 可能需要修改
            if len(LT_max_arr) == 1:
                # 给节点赋予最相似的候选节点的标签
                labels[u] = labels[LT_max_arr[0]]
            else:
                n = np.random.randint(0, len(LT_max_arr))
                labels[u] = labels[LT_max_arr[n]]

    communities = {}
    for key, val in labels.items():
        if val in communities:
            communities[val].append(key)
        else:
            communities[val] = [key]

    subG_arr = []
    for index, c in communities.items():
        subG_arr.append((c, generate_subG(G, c)))

    SPC = {}
    for (c, communicability) in subG_arr:

        for key, val in communicability.items():
            spc = 0
            for inx, value in val.items():
                if inx != key:
                    spc += value
            SPC[key] = spc

    LI = {}
    for u in G.nodes():
        for (c, comm) in subG_arr:
            if u in c:
                li = 0
                c_copy = deepcopy(c)
                c_copy.remove(u)
                for v in c_copy:
                    li += transforms(SPC[u], SPC[v])
                LI[u] = li

    P = [node for (node, val) in LI.items() if val == 0 and node not in candidate]

    cs = P + candidate
    return cs


def generation_of_seed_set(G, cs, k, SP, candidate):
    S = []
    cnt = 1

    cs_d = {}
    for (key, val) in SP:
        if key in cs:
            cs_d[key] = val
    cs = [i for i in sorted([(i, v) for i, v in cs_d.items()], key=lambda x: x[1], reverse=True)]

    S_1 = cs[:k]

    S_c = [i for (i, v) in S_1]

    S_A = [i for i in candidate if i not in S_c]

    lens = len(S_A) or 1

    while cnt < math.ceil(math.log(lens) * k):
        node_new = S_A[np.random.randint(0, lens - 1)]
        S_p = deepcopy(S_c)
        S_p[np.random.randint(0, len(S_c) - 1)] = node_new

        edv1 = k
        # 计算S_p直接关联的邻居
        N_S1c = []
        for u in S_c:
            for (u, v) in G.edges(u):
                if v not in N_S1c:
                    N_S1c.append(v)
        list_new = [i for i in N_S1c if i not in S_c]

        for u in list_new:
            theta_i = 0
            for (u, v) in G.edges(u):
                if v in S_c:
                    theta_i += 1
            edv1 += 1 - (1 - activation_probability) ** theta_i

        edv2 = k
        # 计算S_p直接关联的邻居
        N_S1p = []
        for u in S_p:
            for (u, v) in G.edges(u):
                if v not in N_S1p:
                    N_S1p.append(v)
        list_new = [i for i in N_S1p if i not in S_p]

        for u in list_new:
            theta_i = 0
            for (u, v) in G.edges(u):
                if v in S_p:
                    theta_i += 1
            edv2 += 1 - (1 - activation_probability) ** theta_i

        gain = edv2 - edv1

        if gain > 0:
            S_c = S_p

        cnt += 1

    S = S_c

    return S

# c = nx.communicability(G)

# x_min = 0
# x_max = 0
# for i, val in c.items():
#     for j, value in val.items():
#         if value < x_min:
#             x_min = value
#         if value > x_max:
#             x_max = value

# k = 10

# candidate, SP = initial_candidate_selection(G, k, c)
# cs = optimization_candidate(G, c, candidate, k)
# print(generation_of_seed_set(G, cs, k, SP, candidate))
