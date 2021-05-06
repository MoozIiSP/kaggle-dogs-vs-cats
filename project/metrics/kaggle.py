import torch
import torch.nn.functional as F
from torchmetrics import Metric


class LogLoss(Metric):
    """Metric for Kaggle dogs vs cats datasets.
    so, y_hat should be [N, 2] because of y is [N, 2] of which values are 0 or 1.
    you should use softmax for y_hat."""

    def __init__(self, dist_sync_on_step=False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("logloss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _compute_loss(self, y_hat, y):
        assert (
            y_hat.shape == y.shape
        ), f"y_hat.shape must be same as y.shape: {y_hat.shape} vs {y.shape}"
        # FIXME: BCELoss dont support FP16 mode 🤷‍♂️. 
        # You should use FP32 mode or just use logits rather than by softmax.
        return F.binary_cross_entropy(y_hat, y.float())

    def update(self, y_hat: torch.Tensor, y: torch.Tensor) -> None:
        loss = self._compute_loss(y_hat, y)
        self.logloss += torch.sum(loss)
        self.total += loss.numel()

    def compute(self) -> torch.Tensor:
        return self.logloss.float() / self.total


class KagglePredRecord(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=dict())

    def update(self, batch_ids, preds):
        assert len(batch_ids) == len(
            preds
        ), f"Length is not equal, {len(batch_ids)} vs {len(preds)}."
        for i, id in enumerate(batch_ids):
            self.preds[id] = preds[i]
