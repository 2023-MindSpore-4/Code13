import mindspore.nn as nn
from mindspore.ops import operations as P
import mindspore.numpy as np
import mindspore.ops.functional as F
import mindspore.ops as ops
from mindspore import nn, Tensor
from mindspore import dtype as mstype 
import mindspore


def KL(alpha, c):
    beta = P.Ones()(c)
    S_alpha = P.ReduceSum(keep_dims=True)(alpha, 1)
    S_beta = P.ReduceSum(keep_dims=True)(beta, 1)
    lnB = P.LGamma()(S_alpha) - P.ReduceSum(P.LGamma()(alpha), axis=1, keep_dims=True)
    lnB_uni = P.ReduceSum(P.LGamma()(beta), axis=1, keep_dims=True) - P.LGamma()(S_beta)
    dg0 = ops.Digamma()(S_alpha)
    dg1 = ops.Digamma()(alpha)
    kl = P.ReduceSum((alpha - beta) * (dg1 - dg0), axis=1, keep_dims=True) + lnB + lnB_uni
    return kl


# Using the Expected Cross Entropy
def ce_loss(p, alpha, c, global_step, annealing_step):
    S = P.ReduceSum(alpha, 1)
    E = alpha - 1
    print("p的类型",p.dtype)
    on_value, off_value = Tensor(1, mindspore.int32), Tensor(0, mindspore.int32)
    label = F.one_hot(p, depth=c, on_value=on_value, off_value=off_value, axis=-1)
    A = P.ReduceSum(label * (ops.Digamma()(S) - ops.Digamma()(alpha)), axis=1, keep_dims=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c) 
    return (A + B)




def mse_loss(p, alpha, c, global_step, annealing_step=1):
    S = P.ReduceSum(alpha, 1)
    E = alpha - 1
    m = alpha / S
    label = F.one_hot(p, num_classes=c)
    A = P.ReduceSum((label - m) ** 2, axis=1, keep_dims=True)
    B = P.ReduceSum(alpha * (S - alpha) / (S * S * (S + 1)), axis=1, keep_dims=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    C = annealing_coef * KL(alp, c)
    return (A + B) + C

class UGC(nn.Cell):
    def __init__(self, classes, views, classifier_dims, annealing_epochs=1):
        """
        :param classes: 数据类别个数
        :param views: 数据视图个数
        :param classifier_dims: 分类器的维度
        :param annealing_epoch: 可信分类网络中的超参数
        """
        super(UGC, self).__init__()
        # 定义和初始化一些参数与网络
        self.views = views
        self.classes = classes
        self.annealing_epochs = annealing_epochs
        self.Net = nn.CellList([Net(classifier_dims[i], classes) for i in range(self.views)])
    

    def DS_Combin(self, alpha):
        """
        使用DS证据理论结合多个狄利克雷分布控制的主观意见的函数
        :param alpha: 所有狄利克雷分布的参数
        :return: 组合后的狄利克雷分布参数
        """
        def DS_Combin_two(alpha1, alpha2):
            """
            Dempster’s  combination  rule  for  two  independent  sets  of  masses
            :param alpha1: 第一个视角的狄利克雷分布参数
            :param alpha2: 第二个视角的狄利克雷分布参数
            :return: 组合后的狄利克雷分布参数
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = np.sum(alpha[v], axis=1, keepdims=True)
                E[v] = alpha[v] - 1
                b[v] = E[v] / (S[v].broadcast_to(E[v].shape))
                u[v] = self.classes / S[v]

            # b^0 @ b^(0+1)
            bb = np.matmul(b[0].reshape(-1, self.classes, 1), b[1].reshape(-1, 1, self.classes))
            # b^0 * u^1
            uv1_expand = u[1].broadcast_to(b[0].shape)
            bu = np.multiply(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].broadcast_to(b[0].shape)
            ub = np.multiply(b[1], uv_expand)
            # calculate C
            bb_sum = np.sum(bb, axis=(1, 2))
            bb_diag = np.diagonal(bb, axis1=-2, axis2=-1).sum(-1)
            # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (np.multiply(b[0], b[1]) + bu + ub) / ((1 - C).reshape(-1, 1).broadcast_to(b[0].shape))
            # calculate u^a
            u_a = np.multiply(u[0], u[1]) / ((1 - C).reshape(-1, 1).broadcast_to(u[0].shape))
            # test = torch.sum(b_a, dim = 1, keepdim = True) + u_a #Verify programming errors

            # calculate new S
            S_a = self.classes / u_a
            # calculate new e_k
            e_a = np.multiply(b_a, S_a.broadcast_to(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        alpha_a = alpha[0]
        for v in range(len(alpha)-1):
            alpha_a = DS_Combin_two(alpha_a, alpha[v+1])
        return alpha_a

    def construct(self, input, y, global_step, batch_idx, sn, if_test=0):
        if(if_test):
            evidence, sigma = self.collect(input)
            alpha = []
            for v_num in range(self.views):
                alpha.append(evidence[v_num] + 1)
            alpha_a = self.DS_Combin(alpha)
            evidence_a = alpha_a - 1
            return evidence_a
        # 分类器损失
        evidence, sigma = self.collect(input)
        loss_class, loss_uc = 0, 0
        alpha = []
        for v_num in range(self.views):
            sn_v = ops.expand_dims(sn[:, v_num], axis=1)
            alpha.append(evidence[v_num] + 1)
            _ = ce_loss(y, alpha[v_num], self.classes, global_step, self.annealing_epochs)
            loss_class += _
        alpha_a = self.DS_Combin(alpha)
        evidence_a = alpha_a - 1

        loss_class +=  ce_loss(y, alpha_a, self.classes, global_step, self.annealing_epochs)

        loss = loss_class.mean(axis=0)

        
        return evidence_a, loss
            

    def collect(self, input):
        """
        :param input: Multi-view data
        :return: evidence of every view
        """
        evidence, sigma = [], []
        for v_num in range(self.views):
            eve, sig = self.Net[v_num](input[v_num])
            evidence.append(eve)
            sigma.append(sig)
        return evidence, sigma
    
    
class SoftPlus(nn.Cell):
    def __init__(self):
        super(SoftPlus, self).__init__()
        self.exp = P.Exp()
        self.log = P.Log()

    def construct(self, x):
        return self.log(self.exp(x) + 1)
    
    
class Net(nn.Cell):
    def __init__(self, classifier_dims, classes):
        super(Net, self).__init__()
        self.classes = classes
        self.fc = nn.SequentialCell([
                  nn.Dense(classifier_dims[0], 64),
                  nn.Sigmoid()
        ])

        self.evidence = nn.SequentialCell([
                  nn.Dense(64, classes),
                  SoftPlus()
        ])

        self.sigma = nn.SequentialCell([
                  nn.Dense(64, 1),
                  SoftPlus()
        ])


    def construct(self, x):

        out = self.fc(x)

        evidence = self.evidence(out)

        sigma = self.sigma(out)

        return evidence*evidence, sigma
