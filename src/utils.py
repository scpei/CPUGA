import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import tensorflow as tf
import math
import random
flags = tf.app.flags
FLAGS = flags.FLAGS


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, placeholders):
    """Construct feed dictionary for GCN-Align."""
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def loadfile(fn, num=1):
    """Load a file and return a list of tuple containing $num integers in each line."""
    print('loading a file...' + fn)
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret


def get_ent2id(fns):
    ent2id = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                ent2id[th[1]] = int(th[0])
    return ent2id


def loadattr(fns, e, ent2id):
    """The most frequent attributes are selected to save space."""
    cnt = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] not in ent2id:
                    continue
                for i in range(1, len(th)):
                    if th[i] not in cnt:
                        cnt[th[i]] = 1
                    else:
                        cnt[th[i]] += 1
    fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
    num_features = min(len(fre), 2000)
    attr2id = {}
    for i in range(num_features):
        attr2id[fre[i][0]] = i
    M = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] in ent2id:
                    for i in range(1, len(th)):
                        if th[i] in attr2id:
                            M[(ent2id[th[0]], attr2id[th[i]])] = 1.0
    row = []
    col = []
    data = []
    for key in M:
        row.append(key[0])
        col.append(key[1])
        data.append(M[key])
    return sp.coo_matrix((data, (row, col)), shape=(e, num_features)) # attr


def get_dic_list(e, KG):
    M = {}
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        M[(tri[0], tri[2])] = 1
        M[(tri[2], tri[0])] = 1
    dic_list = {}
    for i in range(e):
        dic_list[i] = []
    for pair in M:
        dic_list[pair[0]].append(pair[1])
    return dic_list


def func(KG):
    head = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            head[tri[1]] = set([tri[0]])
        else:
            cnt[tri[1]] += 1
            head[tri[1]].add(tri[0])
    r2f = {}
    for r in cnt:
        r2f[r] = len(head[r]) / cnt[r]
    return r2f


def ifunc(KG):
    tail = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            tail[tri[1]] = set([tri[2]])
        else:
            cnt[tri[1]] += 1
            tail[tri[1]].add(tri[2])
    r2if = {}
    for r in cnt:
        r2if[r] = len(tail[r]) / cnt[r]
    return r2if


def get_weighted_adj(e, KG):
    r2f = func(KG)
    r2if = ifunc(KG)
    M = {}
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = max(r2if[tri[1]], 0.3)
        else:
            M[(tri[0], tri[2])] += max(r2if[tri[1]], 0.3)
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = max(r2f[tri[1]], 0.3)
        else:
            M[(tri[2], tri[0])] += max(r2f[tri[1]], 0.3)
    row = []
    col = []
    data = []
    for key in M:
        row.append(key[1])
        col.append(key[0])
        data.append(M[key])
    return sp.coo_matrix((data, (row, col)), shape=(e, e))


def get_ae_input(attr):
    return sparse_to_tuple(sp.coo_matrix(attr))


def load_data(dataset_str):
    names = [['ent_ids_1', 'ent_ids_2'], ['training_attrs_1', 'training_attrs_2'], ['triples_1', 'triples_2'], ['ref_ent_ids']]
    for fns in names:
        for i in range(len(fns)):
            fns[i] = '../data/'+dataset_str+'/'+fns[i]
    Es, As, Ts, ill = names
    ill = ill[0]
    e = len(set(loadfile(Es[0], 1)) | set(loadfile(Es[1], 1)))
    ILL = loadfile(ill, 2)
    np.random.shuffle(ILL)
    KG = loadfile(Ts[0], 3) + loadfile(Ts[1], 3)
    adj = get_weighted_adj(e, KG) # nx.adjacency_matrix(nx.from_dict_of_lists(get_dic_list(e, KG)))

    test = load_train_test_into_model('test.txt')
    test = test.tolist()

    new_train = load_train_test_into_model('new_train.txt')

    train = new_train

    train = train.tolist()
    train_truth = random.sample(list(train[:int(len(train)*0.6)]), int(len(train)//3))
    train_truth = np.asarray(train_truth)
    train_noise = [item for item in train if item not in train_truth]
    train_noise = np.asarray(train_noise)
    train = np.asarray(train)
    unlabeled = load_train_test_into_model('unlabeled.txt')

    return adj, e, train, test, train_truth, train_noise, unlabeled

def save_train_test_into_txt(z, fname):
    np.savetxt('../data/zh_en/' + fname, z, fmt='%d')

def load_train_test_into_model(fname):
    return np.loadtxt('../data/zh_en/' + fname, dtype=int)


class TrainSet(object):
    def __init__(self, train, neg_samples_size):
        self.len = len(train)
        self.train = [(item[0], item[1]) for item in train.tolist()]
        self.orgin_train = train
        self.neg_samples_size = neg_samples_size
        self.subsampling_weight = {}
        self.pretrain = None
        self.noise = None

    def init_subsampling(self, pretrain):
        for pair in self.train:
            if [pair[0], pair[1]] in pretrain:
                self.subsampling_weight[pair] = 1.0
            else:
                self.subsampling_weight[pair] = 0.0

    def get_pretrain(self, pretrain):
        self.pretrain = [(item[0], item[1]) for item in pretrain]

    def get_noise_involved(self, noise_involved):
        self.noise = [(item[0], item[1]) for item in noise_involved]

    def get_subsampling(self):
        subsampling = []
        for item in self.train:
            subsampling.append(self.subsampling_weight[item])
        return np.asarray(subsampling)
