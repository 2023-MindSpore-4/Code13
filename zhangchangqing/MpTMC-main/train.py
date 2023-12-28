import os
import argparse
import mindspore
from mindspore import context
from mindspore import nn, ops
from mindspore.dataset import GeneratorDataset
from mindspore import Tensor,dataset


from model import TMC
from data import Multi_view_data
import warnings
warnings.filterwarnings("ignore")
argmax = ops.ArgMaxWithValue()

class AverageMeter(object):
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

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'])

    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--lambda-epochs', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='LR',
                        help='learning rate')
    args = parser.parse_args()
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)
    args.data_name = 'handwritten_6views'
    args.data_path = 'datasets/' + args.data_name
    args.dims = [[240], [76], [216], [47], [64], [6]]
    args.views = len(args.dims)


    train_loader = GeneratorDataset(
        Multi_view_data(args.data_path, train=True),  column_names=["view0", "view1", "view2","view3","view4","view5", "target"], shuffle=True)
    train_loader = train_loader.batch(batch_size=args.batch_size)
    test_loader = GeneratorDataset(
        Multi_view_data(args.data_path, train=False),  column_names=["view0", "view1", "view2","view3","view4","view5", "target"], shuffle=False)
    test_loader = test_loader.batch(batch_size=args.batch_size )
    
    N_mini_batches = train_loader.get_dataset_size()
    print('The number of training images = %d' % N_mini_batches)

    model = TMC(10, args.views, args.dims, args.lambda_epochs)
    print(model)
    optimizer = nn.Adam(model.trainable_params(), learning_rate=args.lr, weight_decay=1e-5)

    train_loader = train_loader.create_dict_iterator()
    def train(epoch):
            model.set_train()
            loss_meter = AverageMeter()
            for data in train_loader:
                ndata = dict()
                for v_num in range(len(data)-1):
                    ndata[str(v_num)] = mindspore.Parameter(data["view"+str(v_num)])
                target = mindspore.Parameter(data["target"])
                evidences, evidence_a, loss = model(ndata, target, epoch)
                loss_meter.update(loss.item())
            # for batch_idx, (data, target) in enumerate(train_loader):
            #     for v_num in range(len(data)):
            #         data[v_num] = mindspore.Parameter(data[v_num].cuda())
            #     target = mindspore.Parameter(target.long().cuda())
            #     # refresh the optimizer
               
            #     evidences, evidence_a, loss = model(data, target, epoch)
            #     # compute gradients and take step
                
            #     loss_meter.update(loss.item())

    def test(epoch):
        model.set_train(False)
        loss_meter = AverageMeter()
        correct_num, data_num = 0, 0
        for batch_idx, (data, target) in enumerate(test_loader):
            for v_num in range(len(data)):
                data[v_num] = mindspore.Parameter(data[v_num].cuda())
            data_num += target.size(0)
            target = mindspore.Parameter(target.long().cuda())
            evidences, evidence_a, loss = model(data, target, epoch)
            predicted,_ = argmax(evidence_a.data, 1)
            correct_num += (predicted == target).sum().item()
            loss_meter.update(loss.item())

        print('====> acc: {:.4f}'.format(correct_num/data_num))
        return loss_meter.avg, correct_num/data_num


    for epoch in range(1, args.epochs + 1):
        train(epoch)

    test_loss, acc = test(epoch)
    print('====> acc: {:.4f}'.format(acc))
    