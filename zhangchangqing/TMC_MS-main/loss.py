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
from mindspore.ops import constexpr
import mindspore.context as context


# context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

#装饰器，用于编译时常量计算，求类别数
@constexpr
def construct_tensor(x):
    if x is None:
        raise ValueError("input is an unknown value")
    return Tensor(x,dtype=ms.float32)

class TLCLoss(nn.Cell):
    def __init__(self,cls_num_list=None,max_m=0.5,reweight_epoch=160,reweight_factor=0.05,annealing=500,tau=0.54): #160
        super(TLCLoss,self).__init__()
        self.reweight_epoch = reweight_epoch

        m_list = 1./np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list*(max_m/np.max(m_list))
        m_list = Tensor(m_list,dtype=ms.float32)
        self.m_list = m_list

        if reweight_epoch!=-1:
            idx = 1
            betas = [0,0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.per_cls_weights_enabled = ms.Tensor(per_cls_weights, dtype=ms.float32)
        else:
            self.per_cls_weights_enabled = None
        cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
        C = len(cls_num_list)
        per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor
        per_cls_weights = per_cls_weights / np.max(per_cls_weights)

        # save diversity per_cls_weights
        self.per_cls_weights_enabled_diversity = ms.Tensor(per_cls_weights,dtype=ms.float32)
        self.T = (reweight_epoch+annealing)/reweight_factor
        self.tau = tau
        self.reweight_epoch = reweight_epoch

        #lgamma
        self.lgamma = nn.LGamma()
        #digamma
        self.digamma = nn.DiGamma()
        #onehot
        self.onehot = ops.OneHot()

    def _hook_before_epoch(self,epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch
            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None



    def get_final_output(self,x,y):
        #x = Tensor(128,100)
        #y = Tensor(128,)
        #把y转换成onehot
        index = ops.OneHot()(y,x.shape[1],Tensor(1.0, mindspore.float32), Tensor(0.0, mindspore.float32))

        batch_m = ops.MatMul()(self.m_list[None, :],index.transpose(1, 0))
        batch_m = batch_m.view((-1, 1))
        # x_m = x - 30*batch_m
        x_m = x - 30*batch_m
        return ops.exp(ms.numpy.where(index, x_m, x))


    def construct(self,x,y,epoch,extra_info=None):
        #extra_info = {'num_expert':num_expert,'logits':[Tensor(128,100),Tensor(128,100),Tensor(128,100)],'w':[Tensor(128,),Tensor(128,),Tensor(128,),Tensor(128,)]]}
        #x = Tensor(128,100)
        #y = Tensor(128,)
        loss = 0
        for i in range(extra_info["num_expert"]):
            alpha = self.get_final_output(extra_info["logits"][i], y)
            # alpha =  ops.clip_by_global_norm([alpha],1.0)[0]

            S = alpha.sum(axis=1,keepdims=True)
            l = ops.nll_loss(ops.log(alpha)-ops.log(S),y,weight=self.per_cls_weights_base,reduction="none")
            yi = self.onehot(y,alpha.shape[1],Tensor(1.0, mindspore.float32), Tensor(0.0, mindspore.float32))
            # KL

            # adjusted parameters of D(p|alpha)

            alpha_tilde = yi+(1-yi)*(alpha+1)


            #S_tile --> Tensor(128,1)
            S_tilde = alpha_tilde.sum(axis=1,keepdims=True)

            #静态图中torch.lgamma(torch.tensor(alpha_tilde.shape[1])) == self.lgamma(ms.Tensor(10.0))
            K = construct_tensor(alpha_tilde.shape[1]) #cifar10中为ms.Tensor(10.0)

            kl = self.lgamma(S_tilde)-self.lgamma(K)-self.lgamma(alpha_tilde).sum(axis=1,keepdims=True) \
                +((alpha_tilde-1)*(self.digamma(alpha_tilde)-self.digamma(S_tilde))).sum(axis=1,keepdims=True)
            # l += epoch/self.T*kl.squeeze(-1)


            #  diversity
            if self.per_cls_weights_diversity is not None:
                diversity_temperature = self.per_cls_weights_diversity.view((1,-1))
                temperature_mean = diversity_temperature.mean()
            else:
                diversity_temperature = 1
                temperature_mean = 1
            output_dist = ops.LogSoftmax(axis=1)(extra_info["logits"][i]/diversity_temperature)

            mean_output_dist = ops.Softmax(axis=1)(x/diversity_temperature)
            l -= 0.01*temperature_mean*temperature_mean*ops.KLDivLoss(reduction="none")(output_dist,mean_output_dist).sum(axis=1)
            # dynamic engagement
            w = extra_info['w'][i]/extra_info['w'][i].max()
            w = w>self.tau
            loss += (w*l).sum()/w.sum()
            loss += l.mean()
        return loss

    # nn.LGamma()

