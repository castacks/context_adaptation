from types import SimpleNamespace

import torch

from context_adaptation.models.ExactGP import GPTraversability


class IdentityPredictor:
    def eval(self):
        pass

    def __call__(self, values):
        assert len(values) <= 3
        return SimpleNamespace(mean=values[:, 0], variance=torch.ones(len(values)))


class IdentityLikelihood:
    def eval(self):
        pass

    def __call__(self, prediction):
        return prediction


def test_forward_batches_predictions():
    model = object.__new__(GPTraversability)
    model.gp = IdentityPredictor()
    model.likelihood = IdentityLikelihood()
    model.in_mean = model.out_mean = torch.tensor(0.0)
    model.in_std = model.out_std = torch.tensor(1.0)
    model.prediction_batch_size = 3

    values = torch.arange(7, dtype=torch.float32).view(-1, 1)
    prediction, variance = model.forward(values)

    assert torch.equal(prediction, values[:, 0])
    assert torch.equal(variance, torch.ones(7))
