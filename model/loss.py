from __future__ import absolute_import

import torch
from torch import nn
import torch.nn.functional as F


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as not requiring gradients"


class SequenceCrossEntropyLoss(nn.Module):
    def __init__(self,
                 eos_symbol_id):
        super(SequenceCrossEntropyLoss, self).__init__()
        self.eos_symbol_id = eos_symbol_id

    def forward(self, input, target, length):
        _assert_no_grad(target)
        # length to mask
        batch_size, def_max_length = target.size(0), target.size(1)
        mask = torch.zeros(batch_size, def_max_length)
        for i in range(batch_size):
            end_pos = (target[i] == self.eos_symbol_id).nonzero(as_tuple=True)[0]
            mask[i, :end_pos + 1].fill_(1)
        mask = mask.type_as(input)
        # truncate to the same size
        assert length == input.size(1)
        target = target[:, :length]
        mask = mask[:, :length]
        input = to_contiguous(input).view(-1, input.size(2))
        input = F.log_softmax(input, dim=1)
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target.long()) * mask
        output = torch.sum(output) / batch_size

        return output
