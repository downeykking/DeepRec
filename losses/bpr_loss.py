import torch
import torch.nn as nn


class BPRLoss(nn.Module):
    def __init__(self, reduction="mean"):
        """
        `reduction` (string, optional)
        - Specifies the reduction to apply to the output: `none` | `mean` | `sum`.
        `none`: no reduction will be applied.
        `mean`: the sum of the output will be divided by the number of elements in the output.
        `sum`: the output will be summed.
        Note: size_average and reduce are in the process of being deprecated,
        and in the meantime, specifying either of those two args will override reduction. Default: `sum`
        """
        assert reduction in ["mean", "sum", "none"]
        super().__init__()

        self.reduction = reduction

    def forward(self, users_emb, pos_emb, neg_emb, **kwargs):
        """
        `users_emb` (tensor) - shape (b, d)
        `pos_emb` (tensor) - shape (b, d)
        `neg_emb` (tensor) - shape (b, d)
        """

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)

        # BPR loss
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores))

        # reduction
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            pass
        else:
            raise ValueError("reduction must be  'none' | 'mean' | 'sum'")
        return loss
