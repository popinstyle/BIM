import numpy as np
import pickle
import networkx as nx
import math
import networkx_addon
import pandas as pd
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import joblib
import config

G = nx.read_edgelist(config.dataset, delimiter=',', nodetype = int,  data=(('timestamp', int),))
# trueP = pickle.load(open('./datasets/dolphins/Probability.dic', 'rb'), encoding='latin1')

activation_prob = 0.1


def generate_label(row):
    if row['msi'] > 0.5:
        return 1
    else:
        return 0


def gen_mean(row):
    return (row['CN'] + row['JC'] + row['AA'] + row['KM'] + row['SR']) / 5


def preprocess(G):
    # 获得2-hop间的所有节点
    one_hop = []
    for u in G.nodes():
        for (u, v) in G.edges(u):
            one_hop.append((u, v))

    two_hop = []
    for (u, v) in one_hop:
        for (v, w) in G.edges(v):
            if w != u:
                two_hop.append((u, w))

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
    for (u, v) in tqdm(two_hop):
        common_nodes = nx.common_neighbors(G, u, v)
        CN[(u, v)] = 0
        for w in common_nodes:
            CN[(u, v)] += activation_prob
            CN[(u, v)] += activation_prob

            N_z = len([t for (w, t) in G.edges(w)])
            AA[(u, v)] = math.log(N_z)

        u_neigh = [w for (u, w) in G.edges(u)]
        v_neigh = [w for (v, w) in G.edges(v)]

        # all_comm_nodes = u_neigh + v_neigh

        PA[(u, v)] = len([w for (u, w) in G.edges(u)]) * len([w for (v, w) in G.edges(v)])

        jc[(u, v)] = JC[u][v]
        x = list(KM[1])
        km[(u, v)] = KM[0][list(KM[1]).index(u)][list(KM[1]).index(v)]
        sr[(u, v)] = SR[u][v]

        sources.append(u)
        ends.append(v)

        # long = {}
        # for path in nx.all_simple_paths(G, source=u, target=v):
        #     if len(path) not in long:
        #         long[len(path)] = [path]
        #     else:
        #         long[len(path)].append(path)

        # for (i, arr) in long.items():
        #     KM[(u, v)] += 1 ** i * len(arr)

    data = {'CN': CN, 'JC': jc, 'AA': AA, 'KM': km, 'SR': sr, 'source': sources, 'end': ends}
    df = pd.DataFrame(data, index=two_hop)
    df['msi'] = df.apply(lambda x: gen_mean(x), axis=1)
    df['label'] = df.apply(lambda x: generate_label(x), axis=1)

    return df


def construct_df(G, edge):
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
    for (u, v) in edge:
        u = str(u)
        v = str(v)
        common_nodes = nx.common_neighbors(G, u, v)
        CN[(u, v)] = 0
        for w in common_nodes:
            CN[(u, v)] += activation_prob
            CN[(u, v)] += activation_prob

            N_z = len([t for (w, t) in G.edges(w)])
            AA[(u, v)] = math.log(N_z)

        u_neigh = [w for (u, w) in G.edges(u)]
        v_neigh = [w for (v, w) in G.edges(v)]

        # all_comm_nodes = u_neigh + v_neigh

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

        # long = {}
        # for path in nx.all_simple_paths(G, source=u, target=v):
        #     if len(path) not in long:
        #         long[len(path)] = [path]
        #     else:
        #         long[len(path)].append(path)

        # for (i, arr) in long.items():
        #     KM[(u, v)] += 1 ** i * len(arr)
    for (u, v) in CN:
        if (u, v) not in AA:
            AA[(u, v)] = 0
    edge1 = [(str(u), str(v)) for (u, v) in edge]
    data = {'CN': CN, 'JC': jc, 'AA': AA, 'KM': km, 'SR': sr, 'source': sources, 'end': ends}
    df = pd.DataFrame(data, index=edge1)
    df['msi'] = df.apply(lambda x: gen_mean(x), axis=1)

    return df


def train(X, y):
    grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
             'C': [1, 10, 100, 1000]},
            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    clf = GridSearchCV(
        estimator=SVC(probability=True), param_grid=grid,
        cv=cv, scoring='accuracy', n_jobs=6
    )
    clf.fit(X, y)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    return clf

def main(G):
    df = preprocess(G)
    df.dropna(inplace=True)
    y = df.pop('label')



    X, y = df.to_numpy(), y.to_numpy()

    skf = StratifiedKFold(5, shuffle=True, random_state=42)

    best_params, best_models = [], []

    # for train_idx, test_idx in skf.split(X, y):
    # X_train, X_test = X[train_idx], X[test_idx]
    # y_train, y_test = y[train_idx], y[test_idx]

    model = train(X, y)

        # joblib.dump(model, config.model)

        # pre = model.predict(X_test)

        # if hasattr(model, 'best_paramfgt/"";s_'):
        #     best_params.append(model.best_params_)
        #     best_models.append(model)

    # for key, i in enumerate(best_models):
    joblib.dump(model, config.model)




# main(G)
