import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import combinations
from tqdm import tqdm
from RIC import runRICmodel
import random
import numpy as np
import math
import joblib
import config
from link_prediction import construct_df

undirectG = nx.read_edgelist(config.dataset)


# 以出度大的为起点，向外扩张
def spreadSelectByOutDegree(G, target):
    T = sorted(list(G.nodes()), key=lambda x: G.out_degree(x), reverse=True)
    i = 0
    final_T = []

    for u in T:
        if i >= target:
            break
        if u not in final_T:
            final_T.append(u)
            i += 1
        for (u, v) in G.out_edges(u):
            if i >= target:
                break
            if v not in final_T:
                final_T.append(v)
                i += 1
    final_T = list(set(final_T))

    return final_T


# 对列表的元素进行组合
# 这是算不出来的
# 优化算法 使用贪心算法得到可能的传播范围
def spreadSelectByCombination(G, target):
    all_nodes = list(G.nodes())
    coll = []
    for c in tqdm(combinations(all_nodes, target)):
        # print(len(c), c)
        coll.append(c)

    return coll


def generateSubGraphByDegree(G, coverage):
    nodes = spreadSelectByOutDegree(G, len(list(G.nodes())) * coverage)
    subG = deepcopy(G)
    # 逐一去除节点
    for u in list(subG.nodes()):
        if u not in nodes:
            subG.remove_node(u)
            # pos = nx.shell_layout(subG)
            # nx.draw_networkx(subG, pos, node_color='r', with_labels=True)
            # plt.show()

    return subG


def nodeSpreadByIC(G, u, trueP):
    spreads = []
    # 平均数 搞个蒙特卡洛 模拟100次 或者 50次 向下取整
    for _ in range(50):
        spread = []
        for (v, u) in G.in_edges(u):
            w = G[v][u]['weight']
            if random.random() <= 1 - (1 - trueP[(v, u)]) ** w:
                spread.append(v)
        spreads.append(len(spread))
    return math.ceil(np.mean(spreads))


# 贪心选节点使得最后需要的种子集最小
# 从大图中选择converage的节点，从而跑RIC模型时得到的种子集最小

def greedy(G, coverage, trueP):
    all_A = list(G.nodes())
    temp_G = deepcopy(G)
    target = len(all_A) * coverage
    spread = len(all_A)

    delN = []

    while len(delN) < target:
        node_diffusion = {}
        for u in all_A:
            subG = deepcopy(temp_G)
            subG.remove_node(u)
            node_diffusion[u] = runRICmodel(subG, trueP)
        min_value = min([(key, len(i)) for key, i in node_diffusion.items()], key=lambda x: x[1])[1]

        for i in node_diffusion:
            if len(node_diffusion[i]) == min_value and len(delN) < target:
                temp_G.remove_node(i)
                all_A.remove(i)
                delN.append(i)
                spread -= 1

    T = []
    for u in list(G.nodes()):
        if u not in delN:
            T.append(u)
    return delN


def greedy2(G, coverage, trueP, communities, mid_G, unset_more):
    all_A = list(mid_G.nodes())
    A = list(G.nodes())
    temp_G = deepcopy(mid_G)
    target = len(A) * coverage
    spread = len(mid_G)

    delN = []


    # delN 是从mid_G中生成的

    while len(delN) < target:
        node_diffusion = {}
        for u in tqdm(all_A):
            subG = deepcopy(temp_G)
            subG.remove_node(u)
            S, prob = runRICmodel(subG, trueP)
            # 获取节点对
            node_pair = combinations(S, 2)
            
            # 存在于相同社区的数量
            comm = [[] for i in communities]
            score = 0
            for i in S:
                for j, c in enumerate(communities):
                    if i in c:
                        comm[j].append(i)
                        break
                # 可能存在的连接数
                for j in S:
                    if i != j:
                        try:
                            score += unset_more[str((i, j))][1]
                        except:
                            score += 0
            max_comm = max([len(i) for i in comm])
            node_diffusion[u] = {'seed': S, 'prob': prob,  'comm': max_comm, 'score': (len(S) + max_comm + score)}
        min_value = max([i['score'] for key, i in node_diffusion.items()])

        for i in node_diffusion:
            # 这里有点问题
            if node_diffusion[i]['score'] == min_value and len(delN) < target:
                temp_G.remove_node(i)
                all_A.remove(i)
                delN.append(i)
                spread -= 1


    T = []
    for u in list(G.nodes()):
        if u not in delN:
            T.append(u)
    return delN


def generateSubGraphByGreedy(G, coverage, trueP):
    nodes = greedy(G, coverage, trueP)
    subG = deepcopy(G)
    # 逐一去除节点
    for u in list(subG.nodes()):
        if u not in nodes:
            subG.remove_node(u)
    # pos = nx.shell_layout(subG)
    # nx.draw_networkx(subG, pos, node_color='r', with_labels=True)
    # plt.show()

    return subG


def generateSubGraphByGreedy2(G, coverage, trueP, communities, last_N, unset_more):
    # 为了保存进度，先剔除一遍节点
    # 在上一轮的subG中生成这一轮的subG
    mid_G = deepcopy(G)
    for u in list(mid_G.nodes()):
        if u not in last_N:
            mid_G.remove_node(u)
    nodes = greedy2(G, coverage, trueP, communities, mid_G, unset_more)
    # subG是目标传播范围，那在下一轮如何接着跑呢，
    subG = deepcopy(G)
    # 逐一去除节点
    for u in list(subG.nodes()):
        if u not in nodes:
            subG.remove_node(u)
    print('删减后节点size:', len(list(subG.nodes())))
    # pos = nx.shell_layout(subG)
    # nx.draw_networkx(subG, pos, node_color='r', with_labels=True)
    # plt.show()

    return subG




