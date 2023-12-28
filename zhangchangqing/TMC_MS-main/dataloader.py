import mindspore.dataset as ds
import os
import mindspore
import mindspore as ms
import numpy as np
import matplotlib.pyplot as plt
import pickle
import mindspore.dataset.vision as transforms

from mindspore import ops
from math import *
from mindspore.dataset import Dataset
from PIL import Image

def prepare_trte_data(data_folder, batch_size):



    # Do any necessary preprocessing or augmentation here

    train_trsfm = mindspore.dataset.transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((32, 32), (4, 4, 4, 4)),
            transforms.Resize(32),
            transforms.Rescale(1.0 / 255.0, 0.0),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            transforms.HWC2CHW()])


    # mean = {
    #     'cifar10': (0.4914, 0.4822, 0.4465),
    #     'cifar100': (0.5071, 0.4867, 0.4408),
    # }
    #
    # std = {
    #     'cifar10': (0.2023, 0.1994, 0.2010),
    #     'cifar100': (0.2675, 0.2565, 0.2761),
    # }


    test_trsfm = mindspore.dataset.transforms.Compose([
            transforms.Resize(32),
            transforms.Rescale(1.0 / 255.0, 0.0),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            transforms.HWC2CHW()])

    target_trans = mindspore.dataset.transforms.TypeCast(ms.int32)


    train_loader = ds.Cifar10Dataset(dataset_dir=data_folder,usage='train',shuffle=True,num_parallel_workers=2)
    test_loader = ds.Cifar10Dataset(dataset_dir=data_folder,usage='test',shuffle=True,num_parallel_workers=2)

    #得到train_loader所有data

    #数据增强
    train_loader = train_loader.map(train_trsfm, 'image', num_parallel_workers=2)
    test_loader = test_loader.map(test_trsfm, 'image', num_parallel_workers=2)

    train_loader = train_loader.map(operations=target_trans,input_columns='label',num_parallel_workers=2)
    test_loader = test_loader.map(operations=target_trans, input_columns='label', num_parallel_workers=2)


    train_loader = train_loader.batch(4096, drop_remainder=False)
    test_loader = test_loader.batch(4096, drop_remainder=False)

    # cls_num_list = [5000 for _ in range(10)]
    #
    # return train_loader, test_loader, cls_num_list


    cls_num_list = get_cls_num_list()
    train_loader_new = gen_imbalanced_data(train_loader, cls_num_list)
    train_loader_new = train_loader_new.batch(batch_size, drop_remainder=False)
    return train_loader_new, test_loader, cls_num_list




def get_cls_num_list(num_class=10, decay_stride=2.1971,img_max=5000):

    img_num_per_cls = []
    for cls_idx in range(num_class):
        num = img_max * exp(-cls_idx / decay_stride)
        img_num_per_cls.append(int(num + 0.5))
    cls_num_list = img_num_per_cls
    return cls_num_list
def gen_imbalanced_data(train_dataset, img_num_per_cls):
    img_max = 5000
    new_data, new_targets = [], []
    #定义data
    data = np.zeros((0, 3, 32, 32))
    targets_np = np.zeros(0)
    for batch, (X, y) in enumerate(train_dataset.create_tuple_iterator()):
        #将X在第0维上拼接
        data = np.concatenate((X.asnumpy(),data),axis=0)
        targets_np = np.concatenate((y.asnumpy(),targets_np),axis=0)


    classes = np.arange(10)

    num_per_cls = np.zeros(10)
    for class_i, volume_i in zip(classes, img_num_per_cls):
        num_per_cls[class_i] = volume_i
        idx = np.where(targets_np == class_i)[0]
        np.random.shuffle(idx)
        keep_num = volume_i
        selec_idx = idx[:keep_num]
        new_data.append(data[selec_idx, ...])
        new_targets.extend([class_i] * keep_num)
    new_data = np.vstack(new_data)

    train_dataset_new = []
    for i in range(len(new_targets)):
        train_dataset_new.append({'image': new_data[i], 'label': new_targets[i]})

    train_dataset_new = CustomDataset(train_dataset_new)
    train_loader_new = ds.GeneratorDataset(train_dataset_new, column_names=['image', 'label'], shuffle=True)
    return train_loader_new

class CustomDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']

        return image, label