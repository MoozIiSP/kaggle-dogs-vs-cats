import torch
from torch.nn.functional import binary_cross_entropy
from pytorch_lightning.metrics import Metric


class LogLoss(Metric):
    def __init__(self, dist_sync_on_step=False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("logloss", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def _compute_loss(self, y_hat, y, **kwargs):
        criterion_fn = torch.nn.functional.binary_cross_entropy_with_logits
        assert y_hat.shape == y.shape, \
            f"y_hat.shape must be same as y.shape: {y_hat.shape} vs {y.shape}"
        return criterion_fn(y_hat, y.type(torch.float), **kwargs)

    def update(self, y_hat, y):
        loss = self._compute_loss(y_hat, y, reduction='none')
        self.logloss += torch.sum(loss)
        self.total += loss.numel()

    def compute(self):
        return self.logloss.float() / self.total


class KagglePredRecord:
    def __init__(self):
        super().__init__()
        self.preds = dict()

    def update(self, batch_ids, preds):
        assert len(batch_ids) == len(preds), \
            f"Length is not equal, {len(batch_ids)} vs {len(preds)}."
        for i, id in enumerate(batch_ids):
            self.preds[id] = preds[i]
