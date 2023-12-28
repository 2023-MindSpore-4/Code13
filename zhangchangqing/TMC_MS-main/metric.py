import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as ops
import mindspore.ops.operations as P
import mindspore
import numpy as np
import math
import scipy
import pickle
import mindspore.numpy as mnp
from mindspore import Parameter
import mindspore.common.dtype as mstype
from mindspore import Tensor
import mindspore.context as context
from sklearn.metrics import *
from scipy import interpolate

def ACC(output,target,u=None,region_len=100/3):
    pred = ops.argmax(output,axis=1)
    correct = (pred==target)
    region_correct = ms.Tensor((pred/region_len),ms.int64)==ms.Tensor((target/region_len),ms.int64)
    acc = correct.sum(dtype=ms.float32)/len(target)
    region_acc = region_correct.sum(dtype=ms.float32)/len(target)
    split_acc = [0,0,0]

    # count number of classes for each region
    num_class = int(3*region_len)
    region_idx = (ops.arange(num_class)/region_len).astype(ms.int64)
    region_vol = [
        (num_class-ops.count_nonzero(region_idx)).astype(ms.int64),
        ms.Tensor(np.where(region_idx.asnumpy()==1,True,False).sum()),
        ms.Tensor(np.where(region_idx.asnumpy()==2,True,False).sum())
    ]

    target_count = ms.Tensor(np.bincount(target.asnumpy()),ms.int64)
    region_vol = [target_count[:region_vol[0]].sum(dtype=ms.float32), target_count[region_vol[0]:(region_vol[0]+region_vol[1])].sum(dtype=ms.float32),target_count[-region_vol[2]:].sum(dtype=ms.float32)]
    for i in range(len(target)):
        split_acc[region_idx[target[i]]] += correct[i]
    split_acc = [split_acc[i]/region_vol[i] for i in range(3)]

    print('Classification ACC:')
    print('\t all \t =',acc)
    print('\t region  =',region_acc)
    print('\t head \t =',split_acc[0])
    print('\t med \t =',split_acc[1])
    print('\t tail \t =',split_acc[2])
    return acc, region_acc, split_acc[0], split_acc[1], split_acc[2]
