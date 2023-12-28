import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from mindspore.dataset import GeneratorDataset



class Multi_view_data:
    """
    load multi-view data
    """

    def __init__(self, root, train=True):
        """
        :param root: data name and path
        :param train: load training set or test set
        """
        # super(Multi_view_data, self).__init__()
        self.root = root
        self.train = train
        data_path = self.root + '.mat'

        dataset = sio.loadmat(data_path)
        view_number = int((len(dataset) - 5) / 2)

        self.__data = dict()
        if train:
            for v_num in range(view_number):
                self.__data[v_num] = normalize(dataset['x' + str(v_num + 1) + '_train'])
            y = dataset['gt_train']
        else:
            for v_num in range(view_number):
                self.__data[v_num] = normalize(dataset['x' + str(v_num + 1) + '_test'])
            y = dataset['gt_test']
       
        if np.min(y) == 1:
            y = y - 1
        tmp = np.zeros(y.shape[0])
        y = np.reshape(y, np.shape(tmp))
        self.__lable = y


    def __getitem__(self, index):
        # data = dict()
        # for v_num in range(len(self.__data)):
        #     data[v_num] = (self.__data[v_num][index]).astype(np.float32)
        # for v_num in range(len(self.__data)):
        #     data.append(self.__data[v_num][index].astype(np.float32))
        view0 = (self.__data[0][index]).astype(np.float32)
        view1 = (self.__data[1][index]).astype(np.float32)
        view2 = (self.__data[2][index]).astype(np.float32)
        view3 = (self.__data[3][index]).astype(np.float32)
        view4 = (self.__data[4][index]).astype(np.float32)
        view5 = (self.__data[5][index]).astype(np.float32)
        
        target = self.__lable[index]
        # data = np.array(data)
        return view0, view1, view2, view3, view4, view5, target

    def __len__(self):
        print(len(self.__data[0]))
        return len(self.__data[0])


def normalize(x, min=0):
    if min == 0:
        scaler = MinMaxScaler([0, 1])
    else:  # min=-1
        scaler = MinMaxScaler((-1, 1))
    norm_x = scaler.fit_transform(x)
    return norm_x
