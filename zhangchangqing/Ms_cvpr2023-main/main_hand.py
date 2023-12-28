import os
import numpy as np
import mindspore
from mindspore import nn, Tensor
from mindspore.train import Model
from EarlyStopping_hand import EarlyStopping
from mindspore.train.serialization import load_checkpoint
from mindspore import context
from mindspore import ops
from mindspore.nn import Momentum
from model import UGC
from util import mv_dataset, read_mymat, build_ad_dataset, process_data,  mv_tabular_collate,  partial_mv_dataset, partial_mv_tabular_collate
from collections import Counter
from select_k_neighbors import get_samples
import mindspore.dataset as ds
import mindspore as ms
from mindspore import dtype as mstype    
from mindspore.nn import WithLossCell, TrainOneStepCell
from mindspore.nn.optim import Adam
from mindspore.ops import operations as P
import mindspore.context as context



context.set_context(mode=context.GRAPH_MODE, device_target="CPU", save_graphs=False)

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == "__main__":
    import argparse

    # 输入一些参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--pretrain_epochs', type=int, default=0, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--annealing-epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate [default: 1e-3]')
    parser.add_argument('--patience', type=int, default=30, metavar='LR',
                        help='learning rate [default: 1e-3]')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--missing-rate', type=float, default=0, metavar='LR',
                        help='missingrate [default: 0]')
    parser.add_argument('--k-sample', type=int, default=30, metavar='LR',
                        help='times of sampling [default: 10]')
    parser.add_argument('--k', type=int, default=10, metavar='LR',
                        help='number of neighbors [default: 0]')
    parser.add_argument('--k_test', type=int, default=10, metavar='LR',
                        help='number of neighbors [default: 0]')
    parser.add_argument('--slices', type=int, default=1, metavar='LR',
                        help='number of slices [default: 1]')
    parser.add_argument('--if-mean', type=int, default=0, metavar='LR',
                        help='if mean [default: True]')
    parser.add_argument('--latent-dim', type=int, default=64, metavar='LR',
                        help='if mean [default: True]')
    args = parser.parse_args()

    args.decoder_dims = [[240], [76], [216], [47], [64], [6]]
    args.encoder_dims = [[240], [76], [216], [47], [64], [6]]
    args.classifier_dims = [[240], [76], [216], [47], [64], [6]]
    view_num=6

    # 读取数据和划分训练验证以及测试集
    dataset_name = 'handwritten0.mat'
    missing_rate = args.missing_rate
    X, Y, Sn = read_mymat('./data/', dataset_name, ['X', 'Y'], missing_rate)
    partition = build_ad_dataset(Y, p=0.8, seed=999)
    dist_path = './data/hand-dist.mat'

    X = process_data(X, view_num)
    X_train, Y_train, X_test, Y_test, Sn_train = get_samples(x=X, y=Y, sn=Sn,
                                                   train_index=partition['train'],
                                                   test_index=partition['test'],
                                                   dist_path = dist_path,
                                                   n_sample=args.k_sample,
                                                   k=args.k)
    data_train = []
    x_train = X_train[0]
    for i in range(1, len(X_train)):
        x_train = np.concatenate((x_train, X_train[i]),axis=1)
    for i in range(X_train[0].shape[0]):
        # 取出每个特征中对应行的数据并装入列表中
        x_row = np.array(x_train[i])[np.newaxis, :]
        #x_row = np.array(X_train[0][i])[np.newaxis, ...]
        s_row = np.array(Sn_train[i])[np.newaxis, ...]
        y_row = np.array(Y_train[i])
        data_train.append((x_row, s_row, y_row))
    
    ds_train = ds.GeneratorDataset(data_train, ['data', 'sn', 'label'], num_parallel_workers=8)

    train_loader = ds_train.shuffle(buffer_size=120).batch(args.batch_size, drop_remainder=True)
    train_iter = train_loader.create_dict_iterator()
    
    data_test = []
    x_test = X_test[0]
    for i in range(1, len(X_test)):
        x_test = np.concatenate((x_test, X_test[i]),axis=1)
    for i in range(X_test[0].shape[0]):
        # 取出每个特征中对应行的数据并装入列表中
        x_row = np.array(x_test[i])[np.newaxis, :]
        y_row = np.array(Y_test[i])
        data_test.append((x_row, y_row))
    
    ds_test = ds.GeneratorDataset(data_test, ['data', 'label'], num_parallel_workers=8)

    test_loader = ds_test.batch(args.k_sample, drop_remainder=True)
    test_iter = test_loader.create_dict_iterator()
    


    # 构建TMC模型
    context.set_context(mode=context.GRAPH_MODE)
    model = UGC(10, 6, args.classifier_dims, args.annealing_epochs)
    params = model.trainable_params()
    optimizer = Adam(params, learning_rate=args.lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-5)
    
    def train(epoch):
        model.set_train()
        loss_meter = AverageMeter()
        for batch_idx, (data_, sn, target) in enumerate(train_loader):
            data = []
            data.append(data_[:,:, :240])
            data.append(data_[:,:, 240:316])
            data.append(data_[:,:, 316:532])
            data.append(data_[:,:, 532:579])
            data.append(data_[:,:, 579:643])
            data.append(data_[:,:, 643:649])


            for v_num in range(len(data)):
                data[v_num] = np.squeeze(data[v_num], axis=1)
                data[v_num] = Tensor(data[v_num].astype(np.float32))
            target = Tensor(target.astype(np.int32))
            sn = Tensor(sn.astype(np.int32))
            
            # compute model output
            evidence_a, loss = model(data, target, epoch, batch_idx, sn)
            # compute gradients and take step
            grads = optimizer(loss, model.trainable_params())
            optimizer.apply_gradients(grads)
            loss_meter.update(loss.asnumpy()[0])
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data[0]), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss_meter.sum.asnumpy()/loss_meter.count.asnumpy()))
        return loss_meter.sum / loss_meter.count
    
    # 测试
    def test(epoch, dataloader):
        model.set_train(False)
        data_num, correct_num4 = 0, 0
        for batch_idx, (data_, target) in enumerate(test_loader):
            data = []
            data.append(data_[:,:, :240])
            data.append(data_[:,:, 240:316])
            data.append(data_[:,:, 316:532])
            data.append(data_[:,:, 532:579])
            data.append(data_[:,:, 579:643])
            data.append(data_[:,:, 643:649])


            for v_num in range(len(data)):
                data[v_num] = np.squeeze(data[v_num], axis=1)
                data[v_num] = Tensor(data[v_num].astype(np.float32))
            target = Tensor(target.astype(np.int32))
            data_num += target.shape[0]
            
            with context.set_context(mode=context.GRAPH_MODE):
                evidences = model(data, target, epoch, batch_idx, data, 1)
                # 方法4：采样样本类别投票
                _, predicted = evidences.max(1)
                list_ = predicted.asnumpy().tolist()
                most_ = Counter(list_).most_common(1)[0][0]
                if most_ == target[0]:
                    correct_num4 += 1
        data_num = int(data_num/args.k_sample)
        print(correct_num4, data_num)
        acc4 = correct_num4 / data_num
        print("总样本数：",data_num)
        print('====> 投票：', acc4)
        return acc4

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    for epoch in range(1, args.epochs):
        loss = train(epoch)
        early_stopping(loss*(-1), model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # 加载checkpoint
    ckpt_filename = 'checkpoint_hand.ckpt'
    ckpt_path = os.path.join(os.getcwd(), ckpt_filename)
    load_checkpoint(ckpt_path, net=model)
    acc = test(args.epochs, test_loader)

    with open("./test-hand.txt", "a") as f:
        text = "\t缺失率:" + str(missing_rate) + "\t投票:" + str(acc) +"\n"
        f.write(text)
    f.close()
    
