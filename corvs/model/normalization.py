from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F


class MaskedBatchNorm1d(nn.BatchNorm1d):
    def forward(self, input: torch.FloatTensor, valid_mask: torch.BoolTensor | torch.FloatTensor | torch.IntTensor) -> torch.FloatTensor:  # (batch, ch, time), (batch, time) -> (batch, ch, time)
        """
        Modified from `torch.nn.BatchNorm1d.forward()`.
        This method supports masking.

        Parameters
        ----------
        input : FloatTensor
            Input.
            Shape is (batch, ch, time).
        valid_mask : BoolTensor | FloatTensor | IntTensor
            Mask of valid times.
            True for valid and False for invalid.
            Shape is (batch, time).

        Returns
        -------
        output : FloatTensor
            Normalized output.
            Shape is (batch, ch, time).
        """

        self._check_input_dim(input)
        self._check_mask(valid_mask)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        return self.batch_norm(
            input,
            valid_mask,
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps
        )

    def _check_input_dim(self, input: torch.FloatTensor) -> None:
        if input.dim() != 3:
            raise ValueError(f"expected 3D input (got {input.dim()}D input)")

    def _check_mask(self, mask: torch.BoolTensor | torch.FloatTensor | torch.IntTensor) -> None:
        if mask.dim() != 2:
            raise ValueError(f"expected 2D mask (got {mask.dim()}D mask)")
        if not mask.any():
            raise ValueError("expected non-zero mask")

    @staticmethod
    def batch_norm(
            input: torch.FloatTensor,
            valid_mask: torch.BoolTensor | torch.FloatTensor | torch.IntTensor,
            running_mean: torch.FloatTensor | None,
            running_var: torch.FloatTensor | None,
            weight: Optional[torch.FloatTensor] = None,
            bias: Optional[torch.FloatTensor] = None,
            training: bool = False,
            momentum: float = 0.1,
            eps: float = 1e-5
        ) -> torch.FloatTensor:
        """
        Modified from `torch.nn.functional.batch_norm()`.
        This method supports masking.
        """

        if training:
            F._verify_batch_size(input.size())

        if eps <= 0:
            raise ValueError(f"Eps must be positive, but got {eps}.")

        if training:
            valid_cnt = valid_mask.count_nonzero()
            mean = (valid_mask.unsqueeze(1) * input).sum(dim=(0, 2)) / valid_cnt  # (ch, )
            var = (valid_mask.unsqueeze(1) * (input - mean.unsqueeze(0).unsqueeze(2)) ** 2).sum(dim=(0, 2)) / valid_cnt  # (ch, )
            if running_mean is not None and running_var is not None:
                running_mean.copy_((1 - momentum) * running_mean + momentum * mean)
                running_var.copy_((1 - momentum) * running_var + momentum * valid_cnt / (valid_cnt - 1) * var)
        else:
            mean = running_mean
            var = running_var

        assert mean is not None and var is not None
        output = (input - mean.unsqueeze(0).unsqueeze(2)) / (var + eps).sqrt().unsqueeze(0).unsqueeze(2)
        if weight is not None and bias is not None:
            output = weight.unsqueeze(0).unsqueeze(2) * output + bias.unsqueeze(0).unsqueeze(2)

        return output
