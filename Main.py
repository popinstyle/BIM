import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from random import random
from tqdm import tqdm
import math
import pickle
from spreadSelect import spreadSelectByCombination
from spreadSelect import generateSubGraphByDegree
from RIC import runRICmodel
from spreadSelect import generateSubGraphByGreedy

activation_prob = 0.01

from spreadSelect import greedy as RIMGreedy

# G = nx.read_edgelist('./datasets/dblp/index.txt', nodetype=int, create_using=nx.DiGraph())
G = pickle.load(open('./datasets/dolphins/Small_Final_SubG.G', 'rb'), encoding='latin1')
trueP = pickle.load(open('./datasets/dolphins/Probability.dic', 'rb'), encoding='latin1')
# parameter = pickle.load(open('./datasets/dolphins/Small_nodeFeatures.dic', 'rb'), encoding='latin1')


threshold = {}
for u in G.nodes():
    threshold[u] = np.random.random()

communities = []
overlap_nodes = {}
ran = {}

file_object1 = open("./community/dolphins.txt",'r')
try:
  while True:
      line = file_object1.readline().replace('\r','').replace('\n','')
      if line:
          c = line.split(' ')
          toInt = [int(i) for i in c]
          for i in toInt:
              if i in overlap_nodes:
                  overlap_nodes[i] += 1
              else:
                  overlap_nodes[i] = 1
          communities.append(toInt)
      else:
          break
finally:
    file_object1.close()


coverage = 0.4

# IS2算法中社区里面的节点加和不是全部节点
communities_nodes = []
for c in communities:
    communities_nodes += c
communities_nodes = list(set(communities_nodes))


# 画出重复节点的社区
# for i in nodes:
#     nodes_list = []
#     C = nx.DiGraph()
#     for c in communities:
#         if i in c:
#             nodes_list += c
#     # 构建图
#     print(len(nodes_list))
#     nodes_list = set(nodes_list)
#     print(len(nodes_list))
#     for u in nodes_list:
#         for (u, v) in G.edges(u):
#             if v in nodes_list:
#                 C.add_edge(u, v, weight=activation_prob)
#     nx.spring_layout(C)
#     nx.draw(C, with_labels=True, font_size=16)
#     plt.show()


# 设计一个最大匹配算法，即某几个子数组的长度加起来刚好等于想要的数值，若无满足的将另外的数组拆开添加
# 9月26日暂时就这样了
# overlapping community should be changed
# def maxMatch(communities, target):
#     comm = sorted(communities, key=lambda x: len(x), reverse=True)
#     nodes = []
#     # 输出列表中与这个数字相近的数
#     # 最小值的索引
#     index = 0
#     # 最小的community
#     minimum = 0
#     while target > 0:
#         comm_len = [len(i) for i in comm]
#         index = np.argmin(np.array([(target - i) for i in comm_len]))
#         minimum = comm_len[index]
#         # 考虑另一种情况 如果最后刚好还剩一点差额，如何选择
#         if target - minimum < 0:
#             break
#         nodes += comm[index]
#         target -= minimum
#         del comm[index]
#     # 从下面一个社群中随机选择符合条件的节点
#     rest_nodes = np.random.choice(comm[0], target, replace=False).tolist()
#     nodes += rest_nodes
#     return nodes

def overlapNum(communities, overlap_nodes):
    # 每个社区包含的重叠节点数目
    overlap_nums = []
    for c in communities:
        count = 0
        for i in c:
            if i in overlap_nodes and overlap_nodes[i] > 1:
                count += 1
        overlap_nums.append(count)
    return overlap_nums


def commSortByNum(communities, overlap_nums):
    add_num_comm = []
    for idx, c in enumerate(communities):
        add_num_comm.append([c, overlap_nums[idx]])
    add_num_comm = sorted(add_num_comm, key=lambda x: x[1], reverse=True)
    return add_num_comm


# 按照社区里包含的重叠节点数来排序
def maxMatch2(communities, target, overlap_nodes):
    overlap_nums = overlapNum(communities, overlap_nodes)
    comm = commSortByNum(communities, overlap_nums)
    new_nodes = []
    # 输出列表中与这个数字相近的数
    # 最小值的索引
    index = 0
    # 最小的community
    minimum = 0

    while target > 0:
        comm_len = [len(i[0]) for i in comm]

        index = np.argmin([abs(target - i) for i in comm_len])

        minimum = comm_len[index]
        # bug len里面也有重叠节点，如何减去重叠节点

        # 另一个思路，如果target - i > 0, 从距离最近的社区里选取，删除几个节点后刚好等于target， 如果target - i  < 0 怎么办
        # 选节点时，要注意边的连接情况, 判断社区里节点的连接情况
        # 要找带连边且连边多的点

        # 同一个社区的有非连边，怎么破

        # 考虑另一种情况 如果最后刚好还剩一点差额，如何选择
        # 一般都是大于的情况
        # 现在的问题，如果target较小，且刚开始就小于最小的社区，这种情况下就不管他了，直接等于最小的社区

        # if target - minimum < 0:
            # target_nodes = []
            # for c in comm:
            #     for u in c[0]:
            #         # 这步可以进一步优化，选择具有某些特征的节点，选择和剩余节点有连边的节点
            #         if u not in new_nodes:
            #             if len(target_nodes) == target:
            #                 return new_nodes
            #             new_nodes.append(u)
            #             target_nodes.append(u)

        # 记录一下nodes_len
        prev_len = len(new_nodes)
        new_nodes += comm[index][0]
        new_nodes = list(set(new_nodes))
        node_len = len(new_nodes)

        target -= node_len - prev_len
        del comm[index]

    return new_nodes

# 问题输入：1个图和一个影响范围
# 如：影响到80%的节点种子集的范围在多少

# 随机选择`coverage%`的节点并生成子图
def generateSubGraph(G, coverage, communities, overlap_nodes):
    # nodes = np.random.choice(G.nodes(), int(len(G.nodes()) * coverage), replace=False)
    target_nodes = math.ceil(len(G.nodes()) * coverage)
    nodes = maxMatch2(communities, target_nodes, overlap_nodes)
    # subG = nx.DiGraph()
    subG = deepcopy(G)
    for u in list(subG.nodes()):
        if u not in nodes:
            subG.remove_node(u)
    # for u in nodes:
    #     for (u, v) in G.edges(u):
    #         if v in nodes:
    #             subG.add_edge(u, v, weight=G[u][v]['weight'])

    # pos = nx.shell_layout(subG)
    # nx.draw_networkx(subG, pos, node_color='r', with_labels=True)
    # plt.show()
    return subG

probs = {}


# subG = generateSubGraphByDegree(G, coverage)
# subG = generateSubGraphByGreedy(G, coverage, trueP)

# T = spreadSelectByCombination(G, math.ceil(len(list(G.nodes())) * coverage))
# T = list(subG.nodes())
# reserve_subG = nx.DiGraph()
# for (u, v) in subG.edges():
#     reserve_subG.add_edge(v, u, weight=subG[u][v]['weight'])

# pos = nx.shell_layout(subG)
# nx.draw_networkx(reserve_subG, pos, node_color='r', with_labels=True)
# plt.show()


# 对最终的影响力传播范围排序，通过节点社区重叠的次数
def spreadSortByOverlapNum(overlap_nodes, all_nodes):
    node_list = []
    for u in all_nodes:
        node_list.append([u, overlap_nodes[u]])
    node_list = sorted(node_list, key=lambda x: x[1], reverse=True)
    result_list = [i[0] for i in node_list]
    return result_list

# LT模型最开始所有节点都是激活的
# 有问题还要关注其他节点的激活，
def runRLTmodel(G, trueP):
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

    S = []
    for u in T:
        # 节点阈值
        thres = threshold[u]
        thres_total = 0
        for (v, u) in G.in_edges(u):
            thres_total += trueP[(v, u)]
        # print(thres_total, '节点', u, ':', thres)
        if thres_total >= thres:
            S.append(u)
            remove_node.append(u)
            # G.remove_node(u)
            pos = nx.shell_layout(temp_G)
            nx.draw(G, pos, edge_color="grey", node_size=500, with_labels=True)
            pos_higher = {}
            for k, v in pos.items():  # 调整下顶点属性显示的位置，不要跟顶点的序号重复了
                if v[1] > 0:
                    pos_higher[k] = (v[0] - 0.1, v[1] + 0.1)
                else:
                    pos_higher[k] = (v[0] - 0.1, v[1] - 0.1)
            nx.draw_networkx(temp_G, pos, node_color='r', with_labels=True)
            nx.draw_networkx_nodes(temp_G, pos, nodelist=remove_node, node_color='c')
            nx.draw_networkx_labels(temp_G, pos_higher, labels=current_threshold, font_color="brown", font_size=12)
            nx.draw_networkx_edges(temp_G, pos, edgelist=remove_edge, edge_color="c", arrows=True)
            plt.show()

    filter_T = []
    for u in list(G.nodes()):
        if u not in S:
            filter_T.append(u)
    return filter_T

# LT模型验证
def runLTmodel_n(G, S, trueP, threshold):
    T = deepcopy(S)  # targeted set

    W = dict(zip(G.nodes(), [0] * len(G)))  # weighted number of activated in-neighbors
    Sj = deepcopy(S)
    # print 'Initial set', Sj
    while len(Sj):  # while we have newly activated nodes
        Snew = []
        for u in Sj:
            for (u, v) in G.out_edges(u):
                if v not in T:
                    W[v] += trueP[(u, v)] * G[u][v]['weight']
                    if W[v] >= threshold[v]:
                        Snew.append(v)
                        T.append(v)
        Sj = deepcopy(Snew)


    return T


# # 将子图拆成多条有向边组建，在每个有向边做RIC，接龙，cc
# def runRICmodel2(G, overlap_nodes):



# 验证
def runICmodel_n(G, S, trueP, prob):
    reward = 0
    T = deepcopy(S)
    E = {}
    # print(S)

    remove_node = []
    remove_edge = []
    i = 0
    while i < len(T):
        for (T[i], v) in G.out_edges(T[i]):
            current_edge = []
            w = G[T[i]][v]['weight']
            cur = 0.0
            if (T[i], v) in prob:
                cur = prob[(T[i], v)]
            else:
                cur = random()
            # print(" ".join(map(str, [T[i], v, probs[(T[i],  v)], (1 - (1 - trueP[(T[i], v)]) ** w)])))
            if cur <= 1 - (1 - trueP[(T[i], v)])**w:
                remove_edge.append((T[i], v))
                if v not in T:
                    T.append(v)

                # pos = nx.shell_layout(G)
                # nx.draw_networkx(G, pos, node_color='r', with_labels=True)
                # nx.draw_networkx_nodes(G, pos, nodelist=T, node_color='c')
                # plt.show()
        i += 1
    return T

# 如何加入蒙特卡洛模拟，有指标评价判断吗
# 蒙特卡洛模拟需要更新东西，更新啥呢？
# 如何加入贪心？
# 全覆盖

results = []
seeds = []
# for _ in range(100):
# seed_IC, prob = runRICmodel(subG, trueP)
# print(seed_IC)
# # seed_IC = [36, 52, 62, 56]
# seeds.append(len(seed_IC))
# results.append(len(runICmodel_n(subG, seed_IC, trueP, prob)))

# results = []
# for _ in range(100):
# seed_LT = runRLTmodel(subG, trueP)
# print(seed_LT)
# sp = len(runLTmodel_n(subG, seed_LT, trueP, threshold))
# results.append(sp)
# print('单跑结果种子集', np.mean(seeds))
# print('单跑结果:', np.mean(results))


# 满足coverage的最小种子集
# 每次减去一个节点并计算反向种子集，每次去掉一个使种子集最大的节点，总的spread减去这个节点的spread，直到刚好为target值
# 怎么选到最少的种子节点，且他们的spread可以超过coverage
def greedy(subG, coverage):
    S = []
    temp_G = deepcopy(subG)
    sub_A = list(temp_G.nodes())

    # 计算每个节点的外spread

    target = math.ceil(coverage * len(sub_A))
    # 刚开始是满的spread
    spread = len(sub_A)
    status = True

    # 计算节点的spread， 运用蒙特卡洛模拟

    # 去掉这个节点后的种子集大小为多少
    # while spread > target:
    # 先算一次种子集，减去每个节点后再跑RIC种子集为啥
    diffusion = {}
    temp_diff = []
    del_diff = []
    sub_A = list(temp_G.nodes())
    for u in sub_A:
        sub_subG = deepcopy(temp_G)
        sub_subG.remove_node(u)
        diffusion[u] = runRICmodel(sub_subG, trueP)
    min_node = max(diffusion.items(), key=lambda x: len(x[1]))[0]
    temp_G.remove_node(min_node)

    del_diff.append(min_node)
    rests = {}
    rests[min_node] = diffusion[min_node]
    while len(rests[min_node]) > 10:
        diffusion = {}
        sub_A = list(temp_G.nodes())
        for u in sub_A:
            sub_subG = deepcopy(temp_G)
            sub_subG.remove_node(u)
            diffusion[u] = runRLTmodel(sub_subG, overlap_nodes)

        min_node = max(diffusion.items(), key=lambda x: len(x[1]))[0]
        temp_G.remove_node(min_node)
        del_diff.append(min_node)
        rests[min_node] = diffusion[min_node]

    return del_diff


def nodeSpreadByIC(G, u):
    spreads = []
    # 平均数 搞个蒙特卡洛 模拟100次 或者 50次 向下取整
    for _ in range(10):
        spread = []
        for (v, u) in G.in_edges(u):
            w = G[v][u]['weight']
            if random() <= 1 - (1 - trueP[(v, u)]) ** w:
                spread.append(v)
        spreads.append(len(spread))
    return math.ceil(np.mean(spreads))


def nodeSpreadByLT(G, u):
    spread = 0
    thres = 0
    for (v, u) in G.in_edges(u):
        thres += trueP[(v, u)]
    if thres > threshold[u]:
        spread = G.in_degree(u)
    return spread


def greedy2(G):
    temp_G = deepcopy(G)
    all_A = list(G.nodes())
    spread = len(all_A)
    S = []
    delN = []
    # 用来对比的
    target = len(all_A)
    while len(S) < target / 3:
        node_spread = {}
        seeds = {}
        for u in all_A:
            subG = deepcopy(temp_G)
            subG.remove_node(u)
            seeds[u] = runRICmodel(subG, trueP)
            node_spread[u] = nodeSpreadByIC(G, u)
        min_value = max([(key, i) for key, i in node_spread.items()], key=lambda x: x[1])[1]

        for i in node_spread:
            if node_spread[i] > 0 and len(S) < target / 2:
                delN.append(i)
        S = [item for item in all_A if item not in delN]
        # for i in seeds:
        #     spreadss = runICmodel_n(G, seeds[i])
        #     max_arr.append((i, len(seeds[i]), len(spreadss)))

        # pos = nx.shell_layout(G)
        # nx.draw_networkx(G, pos, node_color='r', with_labels=True)
        # nx.draw_networkx_nodes(G, pos, nodelist=S, node_color='c')
        # plt.show()

    # final_S = []
    # for u in list(G.nodes()):
    #     if u not in S:
    #         final_S.append(u)

    return S



# coverages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# for cover in coverages:
# results = []
S_set = []
# #
# S_greedy = greedy(subG, G, coverage)
# for _ in tqdm(range(100)):
# spread = RIMGreedy(G, coverage, trueP)
# subG = generateSubGraphByGreedy(G, coverage, trueP)
# S = runRICmodel(subG, trueP)
# print(S)
# print(len(runICmodel_n(G, S)))
# S_set.append(len(S_greedy))
# lens = len(runICmodel_n(subG, S_greedy))
# results.append(lens)

# print(np.mean(S_set))
# print(np.mean(results))

#
# results.append(lens)
# print(coverage, len(S_greedy))
# print(len(runICmodel_n(G, S_greedy)))
# print(S_set)
# print(np.mean(results))

# test_G = nx.DiGraph()
# for u in seed:
#     for (u, v) in G.out_edges(u):
#         test_G.add_edge(u, v, weight=round(trueP[(u, v)], 2))

# pos = nx.random_layout(test_G)
# weights = nx.get_edge_attributes(test_G, "weight")
# nx.draw_networkx(test_G, pos, with_labels=True)
# nx.draw_networkx_nodes(G, pos, nodelist=[1], node_color='r')
# nx.draw_networkx_edge_labels(test_G, pos, edge_labels=weights)
#
# plt.show()


# S = [subG.subgraph(c).copy() for c in nx.weakly_connected_components(subG)]
#
# for c in S:
#     print(c)
#     nx.spring_layout(c)
#     nx.draw(c, with_labels=True, font_size=16)
#     plt.show()



# print(len(S))

# nx.spring_layout(subG)
# nx.draw(subG, with_labels=True, font_size=10)
# plt.show()


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
#
# c = girvan_newman(undirectedG.copy())
# node_groups = []
#
# for i in c:
#     node_groups.append(len(l2 ist(i)))
#
#
# print(node_groups)