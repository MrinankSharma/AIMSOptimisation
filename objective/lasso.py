import torch

from objective.base import Objective


class Lasso(Objective):
    def _validate_inputs(self, w, x, y):
        assert w.dim() == 2, \
            "Input w should be 2D"
        assert w.size(1) == 1, \
            "Lasso regression can only perform regression (size 1 output)"
        assert x.dim() == 2, \
            "Input datapoint should be 2D"
        assert y.dim() == 1, \
            "Input label should be 1D"
        assert x.size(0) == y.size(0), \
            "Input datapoint and label should contain the same number of samples"


class Lasso_subGradient(Lasso):
    def task_error(self, w, x, y):
        self._validate_inputs(w, x, y)
        # TODO: Compute mean squared error
        error = torch.mean(torch.pow(torch.mm(x, w) - y.view(-1, 1), 2))
        return error

    def oracle(self, w, x, y):
        self._validate_inputs(w, x, y)
        # regularization hyper-parameter
        mu = self.hparams.mu
        # TODO: Compute objective value
        obj = torch.mean(torch.pow(torch.mm(x, w) - y.view(-1, 1), 2)) + mu/2 * torch.sum(torch.abs(w))
        # TODO: compute subgradient
        N = y.size()[0]
        d = x.size()[1]
        dw = 2/N * torch.mm((torch.mm(x.T, x)), w) - (2/N * torch.mm(x.T, y.view(-1, 1))) + ((mu/2) * torch.sign(w))
        return {'obj': obj, 'dw': dw}


class SmoothedLasso_Gradient(Lasso):
    def task_error(self, w, x, y):
        self._validate_inputs(w, x, y)
        error = torch.mean(torch.pow(torch.mm(x, w) - y.view(-1, 1), 2))
        return error

    def oracle(self, w, x, y):
        self._validate_inputs(w, x, y)
        # regularization hyper-parameter
        mu = self.hparams.mu
        # temperature parameter
        temp = self.hparams.temp
        N = y.size()[0]
        d = x.size()[1]
        ell_1_approx = torch.sum(temp * torch.logsumexp(torch.cat([w, -w], 1) / temp, 1))
        obj = torch.mean(torch.pow(torch.mm(x, w) - y.view(-1, 1), 2)) + mu/2 * ell_1_approx
        # TODO: compute gradient
        dw = 2/N * torch.mm((torch.mm(x.T, x)), w) - (2/N * torch.mm(x.T, y.view(-1, 1))) + mu/2 * torch.tanh(w/temp)
        return {'obj': obj, 'dw': dw}
