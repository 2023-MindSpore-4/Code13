import numpy as np
import mindspore as ms
import random
from train_and_test import train
from time import time

def random_seed_setup(seed: int = None):
    if seed:
        print('Set random seed as', seed)
        ms.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    else:
        pass

def main():
    ms.set_context(device_target="GPU")
    random_seed_setup(666)
    data_folder = 'cifar-10-batches-bin'
    testonly = False
    modelpath = './model/'
    train(data_folder, modelpath, testonly)


if __name__ == '__main__':
    start = time()
    main()
    end = time()
    minute = (end - start) / 60
    hour = minute / 60
    if minute < 60:
        print('Training finished in %.1f min' % minute)
    else:
        print('Training finished in %.1f h' % hour)




