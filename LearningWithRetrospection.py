import numpy as np
import torch
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F
import torch.nn as nn


def soft_crossentropy(logits, y_true, dim):
    return -1 * (torch.log_softmax(logits, dim=dim) * y_true).sum(axis=1).mean(axis=0)


def crossentropy(logits, y_true, dim):
    if dim == 1:
        return F.cross_entropy(logits, y_true)
    else:
        loss = 0.0
        for i in range(logits.shape[1]):
            loss += soft_crossentropy(logits[:, i, :], y_true[:, i, :], dim=1)
        return loss


class LWR(torch.nn.Module):
    def __init__(
        self,
        k: int,
        num_batches_per_epoch: int,
        dataset_length: int,
        output_shape: Tuple[int],
        max_epochs: int,
        tau=5.0,
        update_rate=0.9,
        softmax_dim=1,
        use_kl=False,
    ):
        """
        Args:
            k: int, Number of Epochs after which soft labels are updated (interval)
            num_batches
        """
        super().__init__()
        self.k = k
        self.update_rate = update_rate
        self.max_epochs = max_epochs

        self.step_count = 0
        self.epoch_count = 0
        self.num_batches_per_epoch = num_batches_per_epoch

        self.tau = tau
        self.alpha = 1.0
        self.scaling = 4

        self.softmax_dim = softmax_dim

        self.labels = torch.zeros((dataset_length, *output_shape))
        self.usekl = use_kl

    def forward(
        self,
        batch_idx: Tensor,
        logits: Tensor,
        y_true: Tensor,
        previous_output=None,
        eval=False,
    ):
        self.alpha = 1 - self.update_rate * self.epoch_count * self.k / self.max_epochs
        if self.epoch_count <= self.k:
            self.step_count += 1
            if (
                self.step_count + 1
            ) % self.num_batches_per_epoch == 0 and eval is False:
                self.step_count = 0
                self.epoch_count += 1

            if self.epoch_count == self.k and eval is False:
                # print(self.labels[batch_idx, ...].shape, logits.shape)
                if self.usekl:
                    self.labels[batch_idx, ...] = (
                        torch.softmax(logits / self.tau, dim=self.softmax_dim)
                        .detach()
                        .clone()
                        .cpu()
                    )
            return F.cross_entropy(logits, y_true)
        else:
            if (self.epoch_count + 1) % self.k == 0 and eval is False and use_kl:
                if self.usekl:
                    self.labels[batch_idx, ...] = (
                        torch.softmax(logits / self.tau, dim=self.softmax_dim)
                        .detach()
                        .clone()
                        .cpu()
                    )
            if self.usekl:
                return self.loss_fn_with_kl(logits, y_true, batch_idx)
            else:
                return self.L1_loss_fn(logits, y_true, previous_output)

    def loss_fn_with_kl(
        self, logits: Tensor, y_true: Tensor, batch_idx: Tensor,
    ):
        # assert(logits.shape == y_true.shape)
        return self.alpha * crossentropy(logits, y_true, dim=self.softmax_dim) + (
            1 - self.alpha
        ) * self.tau * self.tau * F.kl_div(
            F.log_softmax(logits / self.tau, dim=self.softmax_dim),
            self.labels[batch_idx, ...].to(logits.get_device()),
            reduction="batchmean",
        )

    def L1_loss_fn(self, logits: Tensor, y_true: Tensor, previous_output: Tensor):
        """
        From Jandial, Surgan, et al. 
        "Retrospective Loss: Looking Back 
        to Improve Training of Deep Neural Networks." 
        Proceedings of the 26th ACM SIGKDD 
        International Conference on Knowledge 
        Discovery & Data Mining. 2020.
        """
        task_loss = crossentropy(logits, y_true, dim=self.softmax_dim)

        logit_final = torch.argmax(logits, dim=1).float()

        b = nn.L1Loss()(logits.detach().cpu(), previous_output.detach().cpu())
        a = nn.L1Loss()(logit_final.detach().cpu(), y_true.detach().cpu())

        retrospective_loss = ((self.scaling + 1) * a) - ((self.scaling) * b)

        return task_loss + retrospective_loss
