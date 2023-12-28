import numpy as np
import mindspore.dataset as ds
import scipy.io as scio
import scipy.sparse as scsp
import mindspore.numpy as mnp
import mindspore.dataset.vision.py_transforms as transforms
import mindspore.nn as nn
from numpy.random import randint
from get_sn import get_sn
from sklearn.preprocessing import StandardScaler




# 输出两个列表X（元素为2维np数组）、Y、Sn
def read_mymat(path, name, sp, missrate, sparse=False):
    '''
    :param k_sample: 采样次数
    :param k: 每个视图的近邻样本数
    '''
    mat_file = path + name
    f = scio.loadmat(mat_file)

    if 'Y' in sp:
        if (name == 'handwritten0.mat') or (name == 'BRAC.mat') or (name == 'ROSMAP.mat'):
            Y = (f['gt']).astype(np.int32)
        else:
            Y = (f['gt']-1).astype(np.int32)
    else:
        Y = None

    if 'X' in sp:
        Xa = f['X']
        Xa = Xa.reshape(Xa.shape[1], )
        X = []
        if sparse:
            for x in Xa:
                X.append(scsp.csc_matrix(x).toarray().astype(np.float64))
        else:
            for x in Xa:
                X.append(x.astype(np.float64))
    else:
        X = None
    n_sample = len(X[0][0])
    n_view = len(X)
    Sn = get_sn(n_view, n_sample, missrate).astype(np.float32)

    for i in range(n_view):
        X[i] = X[i].T
    return X, Y, Sn


# 输出字典{train:索引（列数）,test:索引}
def build_ad_dataset(Y, p, seed=999):
    '''
    Converting the original multi-class multi-view dataset into an anomaly detection dataset
    :param seed: Random seed
    :param p: proportion of normal samples for training
    :param neg_class_id: the class used as negative class (outliers)
    :param Y: the original class indexes
    :return:
    '''
    np.random.seed(seed=seed)
    Y = np.squeeze(Y)
    Y_idx = np.array([x for x in range(len(Y))])
    num_normal_train = np.int_(np.ceil(len(Y_idx) * p))
    train_idx_idx = np.random.choice(len(Y_idx), num_normal_train, replace=False)
    train_idx = Y_idx[train_idx_idx]
    test_idx = np.array(list(set(Y_idx.tolist()) - set(train_idx.tolist())))
    partition = {'train': train_idx, 'test': test_idx}
    return partition


# 数据预处理 均值0方差1
def process_data(X, n_view):
    # 数据预处理
    eps = 1e-10
    # n_view = len(X)
    if (n_view == 1):
        X = StandardScaler().fit_transform(X)
    else:
        X = [StandardScaler().fit_transform(X[i]) for i in range(n_view)]
    return X



class partial_mv_dataset(ds.Dataset):
    def __init__(self, X_train, Sn_train, Y_train):
        self.X_train = X_train
        self.Sn_train = Sn_train
        self.Y_train = Y_train

    def __getitem__(self, index):
        datum = [np.array(self.X_train[v][index][np.newaxis, :]) for v in range(len(self.X_train))]
        Sn = np.array(self.Sn_train[index].reshape(1, -1))
        Y = np.array(self.Y_train[index])
        return {'data': np.concatenate(datum, axis=0), 'sn': Sn, 'label': Y}

    def __len__(self):
        return self.X_train[0].shape[0]
    
    def get_dataset_size(self):
        return len(self.X_train[0])



class mv_dataset(ds.Dataset):
    def __init__(self, data, Y):
        '''
        :param data: Input data is a list of numpy arrays
        '''
        self.data = data
        self.Y = Y

    def __getitem__(self, item):
        datum = [mnp.array(self.data[view][item][np.newaxis, :]) for view in range(len(self.data))]
        Y = mnp.array(self.Y[item])
        return datum, Y

    def __len__(self):
        return self.data[0].shape[0]

def mv_tabular_collate(batch):
    new_batch = [[] for _ in range(len(batch[0][0]))]
    new_label = []
    for y in range(len(batch)):
        cur_data = batch[y][0]
        label_data = batch[y][1]
        for x in range(len(batch[0][0])):
            new_batch[x].append(cur_data[x])
        new_label.append(label_data)
    return [mnp.concatenate(new_batch[i], axis=0) for i in range(len(batch[0][0]))],  mnp.concatenate(new_label, axis=0)

def tensor_intersection(x, y):
    """
    计算两个一维tensor的交集
    :param x:
    :param y:
    :return:
    """
    return mnp.array(list(set(x.asnumpy().tolist()).intersection(set(y.asnumpy().tolist()))))


import mindspore.ops.operations as P

def partial_mv_tabular_collate(batch):
    new_batch = [[] for _ in range(len(batch[0][0]))]
    new_label = []
    new_Sn = []
    for y in range(len(batch)):
        cur_data = batch[y][0]
        Sn_data = batch[y][1]
        label_data = batch[y][2]
        for x in range(len(batch[0][0])):
            new_batch[x].append(cur_data[x])
        new_Sn.append(Sn_data)
        new_label.append(label_data)
    return [P.Concat()(new_batch[i], 0) for i in range(len(batch[0][0]))], P.Concat()(new_Sn, 0), P.Concat()(new_label, 0)
