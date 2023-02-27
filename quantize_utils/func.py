import math
from quantize_utils.quant_utils import *


class QuantFunc(torch.nn.Module):

    def __init__(self):
        super(QuantFunc, self).__init__()


def linear_quant(x, n_nums, min=0., max=1., clamp=True, inplace=False):
    if n_nums == 2:
        # TODO: 1. Test this part. 2. Redesign interface.
        if min != -max:
            raise RuntimeError("Values of min and max don't match.")
        return signer(x, inplace) * max
    # TODO: evaluate the efficiency of normalization.
    scale = n_nums - 1
    if clamp:
        if inplace:
            x.clamp_(min, max)
        else:
            x = x.clamp(min, max)
    if inplace:
        x.sub_(min).div_(max - min)
        affine_quant(x, scale, 0, True)
        x.div_(scale).mul_(max - min).add_(min)
    else:
        x = (x - min) / (max - min)
        x = affine_quant(x, scale, 0, False) / scale
        x = x * (max - min) + min
    return x


class LinearQuant(QuantFunc):

    def __init__(self, n_bits, min=-1., max=1., momentum=0.9):
        super(LinearQuant, self).__init__()
        self.n_bits = n_bits
        self.register_buffer('running_min', torch.FloatTensor([min]))
        self.register_buffer('running_max', torch.FloatTensor([max]))
        self.momentum = momentum

    def forward(self, x):
        if self.n_bits < 32:
            if self.training:
                new_max = x.detach().max()
                new_min = x.detach().min()
                update_with_momentum(self.running_max, new_max, self.momentum)
                update_with_momentum(self.running_min, new_min, self.momentum)
                return linear_quant(x, 1 << self.n_bits, new_min.item(), new_max.item(), clamp=False)
            else:
                return linear_quant(x, 1 << self.n_bits, self.running_min.item(), self.running_max.item(), clamp=True)
        else:
            return x


class QTanh(QuantFunc):

    def __init__(self, n_bits, max=0.1, momentum=0.0):
        super(QTanh, self).__init__()
        self.n_bits = n_bits
        self.register_buffer('running_max', torch.FloatTensor([max]))
        self.momentum = momentum

    def forward(self, x):
        if self.n_bits < 32:
            x = torch.tanh(x)
            if self.training:
                new_max = x.detach().abs().max()
                update_with_momentum(self.running_max, new_max, self.momentum)
                return linear_quant(x, 1 << self.n_bits, -new_max.item(), new_max.item(), clamp=False)
            else:
                return linear_quant(x, 1 << self.n_bits, -self.running_max.item(), self.running_max.item(), clamp=True)
        else:
            return torch.tanh(x)


class QReLU(QuantFunc):

    def __init__(self, n_bits):
        super(QReLU, self).__init__()
        self.n_bits = n_bits

    def forward(self, x):
        if self.n_bits < 32:
            return linear_quant(x, 1 << self.n_bits, 0, 1, clamp=True)
        else:
            return torch.relu(x)


class QLog(QuantFunc):

    def __init__(self, n_bits, eps=1e-11, min=-5, max=0, momentum=0.9):
        super(QLog, self).__init__()
        self.n_bits = n_bits
        self.eps = eps
        self.log2_eps = math.log2(eps)
        self.register_buffer('running_min', torch.FloatTensor([min]))
        self.register_buffer('running_max', torch.FloatTensor([max]))
        self.momentum = momentum

    def forward(self, x):
        if self.n_bits < 32:
            x = torch.log2(x)
            if self.training:
                new_max = x.detach().max()
                new_min = x[x > self.log2_eps].detach().min()
                update_with_momentum(self.running_max, new_max, self.momentum)
                update_with_momentum(self.running_min, new_min, self.momentum)
                linear_quant(x[x > self.log2_eps], (1 << self.n_bits) - 1, new_min.item(), new_max.item(),
                             clamp=False, inplace=True)
            else:
                linear_quant(x[x > self.log2_eps], (1 << self.n_bits) - 1, self.running_min.item(),
                             self.running_max.item(), clamp=True, inplace=True)
            return x
        else:
            return torch.log2(x)


class LogQuant(QuantFunc):

    def __init__(self, n_bits, eps=1e-11, min=0.1, max=1, momentum=0.9, unsigned=False):
        super(LogQuant, self).__init__()
        if unsigned:
            self.log2 = QLog(n_bits, eps, math.log2(min), math.log2(max), momentum)
        else:
            self.log2 = QLog(n_bits - 1, eps, math.log2(min), math.log2(max), momentum)
        self.n_bits = n_bits
        self.unsigned = unsigned

    def forward(self, x):
        if self.n_bits < 32:
            if self.unsigned:
                x = self.log2(x)
                return 2 ** x
            else:
                sign = torch.sign(x)
                x = self.log2(torch.abs(x))
                return sign * (2 ** x)
        else:
            return x


class PACT(QuantFunc):

    def __init__(self, n_bits, alpha=10):
        super(PACT, self).__init__()
        self.n_bits = n_bits
        self.clip_val = torch.nn.Parameter(torch.FloatTensor([alpha]), requires_grad=True)

    def forward(self, x):
        if self.n_bits < 32:
            x = torch.relu(x)
            x = torch.where(x < self.clip_val, x, self.clip_val)
            x = linear_quant(x, 1 << self.n_bits, 0, self.clip_val.item(), clamp=False)
            return x
        else:
            return torch.relu(x)


class WRPN(QuantFunc):

    def __init__(self, n_bits):
        super(WRPN, self).__init__()
        self.n_bits = n_bits

    def forward(self, x):
        if self.n_bits < 32:
            return linear_quant(x, 1 << self.n_bits, -1, 1, clamp=True)
        else:
            return x
