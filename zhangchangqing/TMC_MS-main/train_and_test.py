import os
import mindspore as ms
import mindspore.dataset as ds
import math
import torch
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore import ops
from model import resnet
from mindspore.dataset import Dataset
from mindspore import nn
from dataloader import prepare_trte_data
from mindspore import context
from loss import TLCLoss
from tqdm import tqdm
from metric import ACC


def train(data_folder, modelpath, testonly):
    if 'cifar-10' in data_folder:
        #训练过程使用静态图加速，debug用动态图
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        # context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
        num_epoch = 200
        lr = 0.1
        batch_size = 128
        weight_decay = 2e-4
        num_classes = 10
        num_experts = 3
        gamma = 0.01

        #读入cifar-10-binary文件夹
        current_path = os.path.dirname(__file__)
        data_folder = os.path.join(current_path, data_folder)
        #判断文件夹是否存在
        if os.path.exists(current_path):
            print("data folder exists")
        else:
            print("Wrong data folder")

    else:
        print("Wrong data folder")

    #读取数据
    train_loader,test_loader, cls_num_list = prepare_trte_data(data_folder,batch_size)

    # next(train_loader.create_tuple_iterator())
    #定义模型

    model = resnet([5, 5, 5, 5],
                   num_classes=num_classes,
                   num_experts=num_experts)

    optimizer = nn.SGD(model.trainable_params(), weight_decay=weight_decay, momentum=0.9,
                       nesterov=True)

    #定义损失函数
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = TLCLoss(cls_num_list=cls_num_list)



    if testonly==False:
        for epoch in range(num_epoch):
            train_loop(model, train_loader, loss_fn, epoch, lr, optimizer)
            test_loop(model, test_loader, epoch)

def learning_rate(lr, gamma ,warmup_epoch ,epoch):
    if epoch >= 180:
        factor = gamma * gamma
    elif epoch >= 160:
        factor = gamma
    else:
        factor = 1
    if epoch < warmup_epoch:
        factor = factor * float(1 + epoch) / warmup_epoch
    return lr * factor

def train_loop(model, dataset, loss_fn, epoch, lr, optimizer):
    loss_fn._hook_before_epoch(epoch)


    #改写学习率参数
    ops.assign(optimizer.learning_rate, ms.Tensor(learning_rate(lr=lr,gamma=0.01,warmup_epoch=5,epoch=epoch), ms.float32))
    print("Learning Rate:",optimizer.learning_rate.data.asnumpy())

    # Define forward function
    def forward_fn(data, target):
        output,num_expert,logits,w = model(data)
        extra_info = {
                    "num_expert"    : num_expert  ,
                    "logits"        : logits        ,
                    'w'             : w
                }
        loss = loss_fn(output,target,epoch,extra_info)

        # out = model(data)
        # loss = loss_fn(out[0], target)
        return loss

    # Get gradient function
    grad_fn  = ops.value_and_grad(forward_fn, None, optimizer.parameters)

    # Define function of one-step training
    def train_step(data, target):
        loss, grads = grad_fn(data, target)

        optimizer(grads)
        return loss

    model.set_train(True)
    total_loss = []
    for batch, (data, label) in tqdm(enumerate(dataset.create_tuple_iterator())):
        data = ms.Tensor(data, ms.float32)
        loss = train_step(data, label)
        total_loss.append(loss)
        if batch % 10 == 0:
            print(f'Epoch: {epoch:03d} | Batch: {batch:03d} | Loss: {loss.asnumpy():.4f}')
    print(f'================ Epoch: {epoch:03d} ================')
    print("loss =", sum(total_loss) / len(total_loss))



def test_loop(model, dataset, epoch):
    model.set_train(False)
    OUTPUT = []
    uncertainty = []
    targets = []
    for data, target in dataset.create_tuple_iterator():
    #     out = model(data)
    #     OUTPUT.append(out[0])
    #     targets.append(target)
    # OUTPUT = ms.ops.concat(OUTPUT, 0)
    # targets = ms.ops.concat(targets, 0)
    # print((ops.argmax(OUTPUT,axis=-1) == targets).sum(dtype=ms.float32)*100/len(targets))
        output, num_expert, logits, w = model(data)
        OUTPUT.append(output)
        uncertainty.append(w[-1])
        targets.append(target)
    OUTPUT = ms.ops.concat(OUTPUT, 0)
    uncertainty = ms.ops.concat(uncertainty, 0)
    targets = ms.ops.concat(targets, 0)


    ACC(OUTPUT, targets, uncertainty, region_len=10/3)





