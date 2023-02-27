import torch
import torch.nn.functional as F
from quantize_utils import func as qf


class QConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, quantizer: qf.QuantFunc,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                          padding, dilation, groups, bias)
        self.w_quant = quantizer

    def forward(self, x):
        if self.w_quant is not None:
            w = self.w_quant(self.weight)
        else:
            w = self.weight
        output = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return output
