import config
import pickle
import networkx_addon
import networkx as nx
import math
import pandas as pd
import joblib

# with open(config.dataset, 'r') as file:
#     files = open('../datasets/crime/new.txt', 'w')
#     for line in file:
#         t = line.split(' ')
#         files.write(t[0] + ' ' + t[1] + '\n')
#     files.close()


def generate_label(row):
    if row['msi'] > 0.5:
        return 1
    else:
        return 0


def gen_mean(row):
    return (row['CN'] + row['JC'] + row['AA'] + row['KM'] + row['SR']) / 5


# 生成不存在的边集
G = nx.read_edgelist(config.dataset, delimiter=' ', nodetype = int)

activation_prob = 0.01
unset = []
for u in list(G.nodes()):
    for v in list(G.nodes()):
        if u == v:
            break
        if (u, v) not in G.edges():
            unset.append((u, v))
        if (v, u) not in G.edges():
            unset.append((v, u))


def process(unset):
    CN = {}
    JC = networkx_addon.similarity.jaccard(G)
    PA = {}
    AA = {}
    KM = networkx_addon.similarity.katz(G)
    SR = networkx_addon.similarity.simrank(G)
    jc = {}
    km = {}
    sr = {}
    sources = []
    ends = []
    for (u, v) in unset:
        common_nodes = nx.common_neighbors(G, u, v)
        CN[(u, v)] = 0
        for w in common_nodes:
            CN[(u, v)] += activation_prob
            CN[(u, v)] += activation_prob

            N_z = len([t for (w, t) in G.edges(w)])
            AA[(u, v)] = math.log(N_z)

        PA[(u, v)] = len([w for (u, w) in G.edges(u)]) * len([w for (v, w) in G.edges(v)])
        try:
            jc[(u, v)] = JC[u][v]
        except:
            jc[(u, v)] = 0
        x = list(KM[1])
        km[(u, v)] = KM[0][list(KM[1]).index(u)][list(KM[1]).index(v)]
        sr[(u, v)] = SR[u][v]

        sources.append(u)
        ends.append(v)

    # 添加AA的内容
    for (u, v) in unset:
        if (u, v) not in AA:
            AA[(u, v)] = 0

    data = {'CN': CN, 'JC': jc, 'AA': AA, 'KM': km, 'SR': sr, 'source': sources, 'end': ends}
    df = pd.DataFrame(data, index=unset)
    df['msi'] = df.apply(lambda x: gen_mean(x), axis=1)
    df['label'] = df.apply(lambda x: generate_label(x), axis=1)

    return df


df = process(unset)
# 预测
model = joblib.load(config.model)

y = df.pop('label')
X, y = df.to_numpy(), y.to_numpy()
pre = model.predict(X)
print(pre)

# 存文件，不需要存，直接预测
files = open(config.pred, 'w')
inx = 0
for index, row in df.iterrows():
    files.write(str(index) + '-' + str(y[inx]) + ' ' + str(pre[inx]) + '\n')
    inx += 1


# print(model)