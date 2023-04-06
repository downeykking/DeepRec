import torch
import torch.nn as nn


class EmbLoss(nn.Module):
    def __init__(self, reg_weight=1e-6, norm=2, require_pow=True, reduction="mean"):
        """
            EmbLoss, regularization on embeddings.
            Args:
                reg_weight (float): regularization weight on embeddings.
                norm (int): p-norm of embeddings.
                require_pow (bool): whether power the embeddings' norm.
                reduction (string, optional): Specifies the reduction to apply to the output: `none` | `mean` | `sum`.
                    `none`: no reduction will be applied.
                    `mean`: the sum of the output will be divided by the number of elements in the output.
                    `sum`: the output will be summed.
        """
        super(EmbLoss, self).__init__()
        self.reg_weight = reg_weight
        self.norm = norm
        self.require_pow = require_pow
        self.reduction = reduction

    def forward(self, *embeddings):
        if self.require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(
                    input=torch.norm(embedding, p=self.norm), exponent=self.norm
                )
            emb_loss /= self.norm
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)

        emb_loss *= self.reg_weight

        if self.reduction == "mean":
            emb_loss = torch.mean(emb_loss)
        elif self.reduction == "sum":
            emb_loss = torch.sum(emb_loss)
        elif self.reduction == "none":
            pass
        else:
            raise ValueError("reduction must be  'none' | 'mean' | 'sum'")
        return emb_loss


class EmbMarginLoss(nn.Module):
    """EmbMarginLoss, regularization on embeddings"""

    def __init__(self, power=2):
        super(EmbMarginLoss, self).__init__()
        self.power = power

    def forward(self, *embeddings):
        dev = embeddings[-1].device
        cache_one = torch.tensor(1.0).to(dev)
        cache_zero = torch.tensor(0.0).to(dev)
        emb_loss = torch.tensor(0.0).to(dev)
        for embedding in embeddings:
            norm_e = torch.sum(embedding**self.power, dim=1, keepdim=True)
            emb_loss += torch.sum(torch.max(norm_e - cache_one, cache_zero))
        return emb_loss
