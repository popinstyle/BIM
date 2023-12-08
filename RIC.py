from copy import deepcopy
import random
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

# 问题转化为子图100%覆盖
# 若要得到一个范围，比如30-40个节点，可以采用蒙特卡洛模拟来多次模拟
# 排序 根据出度由低到高排序
# 权值w是一个问题？？？
def runRICmodel(G, trueP):
    # 根据重叠节点出现的次数排序，让节点在每个社群里逐一传播用IC或LT
    # T = spreadSortByOverlapNum(overlap_nodes, list(G.nodes()))
    T = sorted(list(G.nodes()), key=lambda x: G.in_degree(x), reverse=True)
    i = 0
    # 记录一下激活概率值， 假设IC与RIC的激活概率是一致的
    prob = {}
    remove_node = []
    remove_edge = []
    # 反向传播 对所有出度为0的节点，沿着反向有向边做一次反向ic，顺延他们的边，若随机数大于激活概率，去掉入度节点，若小于则保留。
    # 那如何把社区的重叠节点与出度节点结合？？？
    # 添加节点u到S
    delT = []
    temp_G = deepcopy(G)

    # i = 0
    # with open("./file1.txt", 'w') as f:
    while i < len(T):
        for (T[i], u) in list(G.out_edges(T[i])):
            w = 1
            cur = random.random()
            prob[(T[i], u)] = cur
            if cur <= 1 - (1 - trueP[(T[i], u)]) ** w:
                # line = " ".join(map(str, [T[i], u, probs[(T[i], u)], (1 - (1 - trueP[(T[i], u)]) ** w)]))
                # f.write(line + '\n')
                delT.append(u)
                # delT.append(u)
                # pos = nx.shell_layout(G)
                # nx.draw_networkx(G, pos, node_color='r', with_labels=True)
                # nx.draw_networkx_nodes(G, pos, nodelist=delT, node_color='c')
                # plt.show()

        i += 1

    delT = list(set(delT))

    finalS = [u for u in T if u not in delT]

    return finalS, prob