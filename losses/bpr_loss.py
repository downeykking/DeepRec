import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):
    def __init__(self, gamma=1e-10, reduction="mean"):
        """
        Args:
            gamma (float): Small value to avoid division by zero.
            reduction (string, optional): Specifies the reduction to apply to the output: `none` | `mean` | `sum`.
                `none`: no reduction will be applied.
                `mean`: the sum of the output will be divided by the number of elements in the output.
                `sum`: the output will be summed.
        """
        assert reduction in ["mean", "sum", "none"]
        super().__init__()

        self.gamma = gamma
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
        # loss = -F.logsigmoid(pos_scores - neg_scores)
        loss = -torch.log(self.gamma + torch.sigmoid(pos_scores - neg_scores))

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
