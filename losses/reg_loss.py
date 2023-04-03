import torch
import torch.nn as nn


class RegLoss(nn.Module):
    def __init__(self, embed_l2_norm=1e-6, reduction="mean"):
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

        self.embed_l2_norm = embed_l2_norm
        self.reduction = reduction

    def forward(self, users_emb, pos_emb, neg_emb, **kwargs):
        """
        `users_emb` (tensor) - shape (b, d)
        `pos_emb` (tensor) - shape (b, d)
        `neg_emb` (tensor) - shape (b, d)
        """

        # Reg loss
        regularizer = (users_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / 2
        emb_loss = self.embed_l2_norm * regularizer

        # reduction
        if self.reduction == "mean":
            loss = torch.mean(emb_loss)
        elif self.reduction == "sum":
            loss = torch.sum(emb_loss)
        elif self.reduction == "none":
            pass
        else:
            raise ValueError("reduction must be  'none' | 'mean' | 'sum'")
        return loss
