import torch

from objective.base import Objective


class Logistic_Gradient(Objective):
    def _validate_inputs(self, w, x, y):
        assert w.dim() == 2, \
            "Input w should be 2D"
        assert x.dim() == 2, \
            "Input datapoint should be 2D"
        assert y.dim() == 1, \
            "Input label should be 1D"
        assert x.size(0) == y.size(0), \
            "Input datapoint and label should contain the same number of samples"

    def task_error(self, w, x, y):
        self._validate_inputs(w, x, y)
        act_mat = torch.matmul(x, w)
        pred = torch.argmax(act_mat, 1)
        # this throws a user warning for some reason ... annoying
        error = torch.mean(torch.tensor(pred != y).clone().float())
        return error

    def oracle(self, w, x, y):
        self._validate_inputs(w, x, y)
        # regularization hyper-parameter
        mu = self.hparams.mu
        # Compute objective value
        act_mat = torch.matmul(x, w)
        per_point_loss = torch.logsumexp(act_mat, 1).view(-1, 1) - torch.gather(act_mat, 1, y.view(-1, 1))
        # print(per_point_loss)
        obj = torch.mean(per_point_loss) + (mu/2) * (torch.norm(w, p="fro")**2)
        # obj = torch.mean(per_point_loss) + (mu/2) * torch.norm(w, p="fro")**2
        obj.backward()
        dw = w.grad
        return {'obj': obj, 'dw': dw}
