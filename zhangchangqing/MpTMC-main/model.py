import mindspore.nn as nn
import mindspore
import mindspore.numpy as mnp
from mindspore import ops
from mindspore import Tensor,dataset

import mindspore.nn.probability.bijector as msb
import numpy as np

op_sum = ops.ReduceSum(keep_dims=True)
onehot = ops.OneHot(axis=1)
nn_lgamma = nn.LGamma()
nn_digamma = nn.DiGamma()
batmatmul = ops.BatchMatMul()
op_mean = ops.ReduceMean()

def KL(alpha,c):
    beta = ops.ones((1,c),mindspore.float32)
    S_alpha = op_sum(alpha, axis=1)
    S_beta = op_sum(beta, axis=1)

    lnB = nn_lgamma(S_alpha) - op_sum(nn_lgamma(alpha), axis=1)

    lnB_uni = op_sum(nn_lgamma(beta), axis=1) - nn_lgamma(S_beta)
    dg0 = nn_digamma(S_alpha)
    dg1 = nn_digamma(alpha)
    kl = op_sum((alpha - beta) * (dg1 - dg0), axis=1) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = op_sum(alpha, axis=1)
    E = alpha - 1
    depth, on_value, off_value = c, Tensor(1.0, mindspore.float32), Tensor(0.0, mindspore.float32)
    label = onehot(p, depth, on_value, off_value)
    A = op_sum(label * (nn_digamma(S) - nn_digamma(alpha)), axis=1)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    return (A + B)

def mse_loss(p, alpha, c, global_step, annealing_step=1):
    S = op_sum(alpha, axis=1)
    E = alpha - 1
    m = alpha / S
    depth, on_value, off_value = c, Tensor(1.0, mindspore.float32), Tensor(0.0, mindspore.float32)
    label = onehot(p, depth, on_value, off_value)
    A = op_sum((label - m) ** 2, axis=1)
    B = op_sum(alpha * (S - alpha) / (S * S * (S + 1)), axis=1)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    C = annealing_coef * KL(alp, c)
    return (A + B) + C

# a2 = Tensor(alpha,mindspore.dtype.float32)
# p2 = Tensor(p,mindspore.dtype.int32)

class TMC(nn.Cell):

    def __init__(self, classes, views, classifier_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of views
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super().__init__()
        self.views = views
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        self.Classifiers = nn.CellList([Classifier(classifier_dims[i], self.classes) for i in range(self.views)])

    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        def DS_Combin_two(alpha1, alpha2):
            """
            @alpha1: Dirichlet distribution parameters of view 1
            @alpha2: Dirichlet distribution parameters of view 2
            @return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = op_sum(alpha[v], axis=1)
                E[v] = alpha[v]-1
                b[v] = E[v]/(S[v].expand(E[v].shape))
                u[v] = self.classes/S[v]

            # b^0 @ b^(0+1)
            bb = batmatmul(b[0].view(-1, self.classes, 1), b[1].view(-1, 1, self.classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = ops.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = ops.mul(b[1], uv_expand)
            # calculate C
            bb_sum = op_sum(bb, dim=(1, 2), out=None)
            bb_diag = mnp.diagonal(bb, axis1=-2, axis2=-1).sum(-1)
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (ops.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = ops.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = self.classes / u_a
            # calculate new e_k
            e_a = ops.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha)-1):
            if v==0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v+1])
        return alpha_a

    def construct(self, X, y, global_step):
        print('d')
        evidence = self.infer(X)
        print('dd')

        loss = 0
        alpha = dict()
        for v_num in range(len(X)):
            alpha[v_num] = evidence[v_num] + 1
            loss += ce_loss(y, alpha[v_num], self.classes, global_step, self.lambda_epochs)
        alpha_a = self.DS_Combin(alpha)
        evidence_a = alpha_a - 1
        loss += ce_loss(y, alpha_a, self.classes, global_step, self.lambda_epochs)
        loss = op_mean(loss)
        return evidence, evidence_a, loss

    def infer(self, input):
        """
        :param input: Multi-view data
        :return: evidence of every view
        """
        print('xd')
        print(self.Classifiers[0])
        evidence = dict()
        for v_num in range(self.views):
            evidence[v_num] = self.Classifiers[v_num](input[str(v_num)])
        print('evidence',evidence)
        return evidence


class Classifier(nn.Cell):
    def __init__(self, classifier_dims, classes):
        super(Classifier, self).__init__()
        self.num_layers = len(classifier_dims)
        self.fc = nn.CellList()
        for i in range(self.num_layers-1):
            self.fc.append(nn.Dense(classifier_dims[i], classifier_dims[i+1]))
        self.fc.append(nn.Dense(classifier_dims[self.num_layers-1], classes))
        softplus = msb.Softplus()
        self.fc.append(softplus)

    def construct(self, x):
        h = self.fc[0](x)
        for i in range(1, len(self.fc)):
            h = self.fc[i](h)
        return h
