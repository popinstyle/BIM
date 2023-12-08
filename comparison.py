import numpy as np
import networkx as nx
import pickle
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
# from Main import greedy2 as BIMGreedy
from Main import maxMatch2, runICmodel_n, generateSubGraph, runLTmodel_n, runRLTmodel
from baselines.greedy import greedy as IMGreedy
from baselines.celfpp import celfpp
from baselines.ivgreedy import ivgreedy
from baselines.heuristic import single_degree_discount
from baselines.diffusion import IndependentCascade, LinearThreshold
from RIC import runRICmodel
from RLT import runRLTmodel
from spreadSelect import generateSubGraphByGreedy, generateSubGraphByGreedy2
# from temp import process
from copy import deepcopy
from TSIFIM import initial_candidate_selection, optimization_candidate, generation_of_seed_set
from CSR import compute_CSR
import time
import config
import joblib

G = pickle.load(open(config.G, 'rb'), encoding='latin1')

unG = nx.read_edgelist(config.dataset, delimiter=',', create_using=nx.Graph(),  data=(('timestamp', int),))
# G_process = nx.read_edgelist(config.dataset, delimiter=',', nodetype = int)
trueP = pickle.load(open(config.trueP, 'rb'), encoding='latin1')

activation_prob = 0.01
unset = []
for u in list(unG.nodes()):
    for v in list(unG.nodes()):
        if u == v:
            break
        if (u, v) not in unG.edges() and (v, u) not in unG.edges():
            unset.append((u, v))

activation_prob = 0.01
communities = []
overlap_nodes = {}
ran = {}

threshold = {}
for u in G.nodes():
    threshold[u] = np.random.random()

file_object1 = open(config.community, 'r')
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



results = {}
# results['IMGreedy'] = []
# results['CELF++'] = []
# results['ivgreedy'] = []
# results['TSIFIM'] = []
# results['CSR'] = []


# def generateSubGraph(G, nodes):
#     subG = deepcopy(G)
#     # 逐一去除节点
#     for u in list(subG.nodes()):
#         if u not in nodes:
#             subG.remove_node(u)
#     return subG


# subG = generateSubGraphByGreedy(G, 0.2, trueP)
# subG = generateSubGraph(G, 0.2, communities, overlap_nodes)
# T = list(subG.nodes())
# S = runRICmodel(subG, trueP)
# v = 42
prob = {}
# results = []
# for _ in range(50):
#     ori1 = len(runRLTmodel(generateSubGraph(G, T + [v]), trueP, threshold))
#     ori2 = len(runRLTmodel(generateSubGraph(G, T), trueP, threshold))
#     sub1 = len(runRLTmodel(generateSubGraph(G, S + [v]), trueP, threshold))
#     sub2 = len(runRLTmodel(generateSubGraph(G, S), trueP, threshold))
#     results.append((ori1, ori2, sub1, sub2, (ori1 - ori2) <= (sub1 - sub2)))
#
# k = 10
IC = IndependentCascade(G)
LT = LinearThreshold(G)

# for _ in tqdm(range(50)):
#     S = list(celfpp(G, IC, k))
#     T = S + [48]
#     ori1 = len(runICmodel_n(G, T, trueP, prob))
#     ori2 = len(runICmodel_n(G, T, trueP, prob))
#     sub1 = len(runICmodel_n(G, S, trueP, prob))
#     sub2 = len(runICmodel_n(G, S, trueP, prob))
#     results.append((ori1, ori2, sub1, sub2, (ori1 - ori2) <= (sub1 - sub2)))


# c = nx.communicability(unG)
# #
# x_min = 0
# x_max = 0
# for i, val in c.items():
#     for j, value in val.items():
#         if value < x_min:
#             x_min = value
#         if value > x_max:
#             x_max = value


def baselineSelect(name, G, LT, k):
    if name == 'BIMGreedy' or name == 'BIMComm' or name == 'BIMGreedy2':
        return runRICmodel(G, trueP)
    elif name == 'IMGreedy':
        return IMGreedy(G, LT, k)
    elif name == 'CELF++':
        return celfpp(G, LT, k)
    elif name == 'ivgreedy':
        return ivgreedy(G, k)
    elif name == 'TSIFIM':
        return ivgreedy(G, k)
        # candidate, SP = initial_candidate_selection(unG, k, c)
        # cs = optimization_candidate(unG, c, candidate, k, x_max, x_min)
        # return generation_of_seed_set(unG, cs, k, SP, candidate)
    elif name == 'CSR':
        return compute_CSR(G, communities, k)

# k = 1
#
coverage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
coverage_reverse = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

# other algorithms
for alg in results:
    k = 1
    last_N = list(G.nodes())
    for cover in tqdm(coverage):
        starttime = time.time()
        target = math.ceil(len(list(G.nodes())) * cover)
        k = int(cover * len(list(G.nodes()))) - 2
        print('start select seed')
        S = list(baselineSelect(alg, G, LT, k))
        print('start spread')
        spread = len(runICmodel_n(G, S, trueP, prob))
        # spread = len(runLTmodel_n(G, S, trueP, threshold))
        print('spread end', spread)
        # 快速加和
        while spread < target:
            # k += 2
            if target - k < 10:
                k += 1
            else:
                k += 20
            S = list(baselineSelect(alg, G, LT, k))
            spread = len(runICmodel_n(G, S, trueP, prob))
            # spread = len(runLTmodel_n(G, S, trueP, threshold))
            print(k, spread, len(S))
        # 逐步减
        while spread >= target:
            k -= 2
            S = list(baselineSelect(alg, G, LT, k))
            # spread = len(runICmodel_n(G, S, trueP, prob))
            spread = len(runLTmodel_n(G, S, trueP, threshold))
            print(alg, spread, len(S))
        endtime = time.time()
        results[alg].append((len(S), cover, spread, endtime - starttime))
        # print(alg, len(S), target, spread)


# 对BIMGreedy:
results['BIMGreedy'] = []
k = 1
last_N = list(G.nodes())
for cover in tqdm(coverage_reverse):
    unset_more = {}
    with open(config.pred, 'r') as file:
        for line in tqdm(file):
            t = line.split('-')
            y = t[1].split(' ')
            unset_more[t[0]] = (int(y[0]), int(y[1].split('\n')[0]))

    starttime = time.time()
    least_S = []
    least_spread = []
    subG = generateSubGraphByGreedy2(G, cover, trueP, communities, last_N, unset_more)
    last_N = list(subG.nodes())
    S, prob = runRICmodel(subG, trueP)
    # S = runRLTmodel(subG, trueP, threshold)
    least_S.append(len(S))
    # least_spread.append(len(runLTmodel_n(G, S, trueP, threshold)))
    least_spread.append(len(runICmodel_n(subG, S, trueP, prob)))
    endtime = time.time()
    results['BIMGreedy'].append((np.mean(least_S), cover, np.mean(least_spread), endtime - starttime))


r = []
# target = math.ceil(len(list(G.nodes())) * 0.5)
# S = list(baselineSelect('IMGreedy', G, IC, k))
# spread = len(runLTmodel_n(G, S))
# # for _ in range(100):
# while spread < target:
#     k += 1
#     S = list(baselineSelect('IMGreedy', G, IC, k))
#     spread = len(runLTmodel_n(G, S))
#     print(len(S), spread, target)
# # r.append(spread)
# print(S)

# results = {'BIMGreedy': [s(9, 27), (11, 31), (10, 31), (13, 40), (16, 43), (23, 49), (25, 55), (24, 53)], 'BIMComm': [(12, 27), (13, 32), (11, 34), (14, 41), (21, 48), (26, 53), (22, 53), (27, 51)], 'IMGreedy': [(13, 13), (19, 19), (25, 25), (31, 31), (38, 38), (44, 44), (50, 50), (56, 56)], 'celfpp': [(56, 57), (56, 57), (56, 58), (56, 59), (56, 60), (56, 59), (56, 59), (56, 59)]}

seeds_results = {}

# 倒置结果


seeds_results['BIMGreedy'] = [i[0]for i in results['BIMGreedy']][::-1]
# seeds_results['IMGreedy'] = [i[0]for i in results['IMGreedy']]
# seeds_results['CELF++'] = [i[0]for i in results['CELF++']]
# seeds_results['ivgreedy'] = [i[0]for i in results['ivgreedy']]
# seeds_results['TSIFIM'] = [i[0]for i in results['TSIFIM']]
# seeds_results['CSR'] = [i[0]for i in results['CSR']]

time_results = {}
time_results['BIMGreedy'] = [round(i[3],2) for i in results['BIMGreedy']][::-1]
# time_results['IMGreedy'] = [round(i[3],2) for i in results['IMGreedy']]
# time_results['CELF++'] = [round(i[3],2) for i in results['CELF++']]
# time_results['ivgreedy'] = [round(i[3],2) for i in results['ivgreedy']]
# time_results['TSIFIM'] = [round(i[3],2) for i in results['TSIFIM']]
# time_results['CSR'] = [round(i[3],2) for i in results['CSR']]



with open(config.objectfile, 'w') as f:
    for u in results:
        line = str(u) + str(results[u])
        f.write(line + '\n')

# draw seed
f, axa = plt.subplots(1, sharex=True)
inx = 0
line = ['solid', 'dotted', 'dashed', 'dashdot', (0, (5, 5)), (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5))]
marker = [',', '1', 'p', 's', 'x', '8', '*', '|']
for alg_name, result in seeds_results.items():
    axa.plot(coverage, result, label=alg_name, marker=marker[inx])
    for a, b in zip(coverage, result):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=12)
    inx += 1
# 透明背景
axa.legend(bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0, fontsize='x-large')
f.subplots_adjust(right=0.85)

axa.set_xlabel("coverage", fontsize=22)
axa.set_ylabel("seed size", fontsize=22)
axa.set_title("seed size", fontsize=28)
plt.tick_params(labelsize=20)
# plt.savefig('./SimulationResults/AvgReward.pdf', dpi=1200, format='pdf')
plt.show()


# draw running time
f, axa = plt.subplots(1, sharex=True)
inx = 0
line = ['solid', 'dotted', 'dashed', 'dashdot', (0, (5, 5)), (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5))]
marker = [',', '1', 'p', 's', 'x', '8', '*', '|']
for alg_name, result in time_results.items():
    axa.plot(coverage, result, label=alg_name, marker=marker[inx])
    for a, b in zip(coverage, result):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=12)
    inx += 1
# 透明背景
axa.legend(bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0, fontsize='x-large')
f.subplots_adjust(right=0.85)

axa.set_xlabel("coverage", fontsize=22)
axa.set_ylabel("running time(s)", fontsize=22)
axa.set_title("running time(s)", fontsize=28)
plt.tick_params(labelsize=20)
# plt.savefig('./SimulationResults/AvgReward.pdf', dpi=1200, format='pdf')
plt.show()