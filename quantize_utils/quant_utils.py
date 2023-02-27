import torch


class AffineQuantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale, zero_point, inplace):
        ctx.scale = scale
        if inplace:
            ctx.mark_dirty(x)
            x.mul_(scale).sub_(zero_point).round_()
            return x
        else:
            return torch.round(x * scale - zero_point)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None, None, None
        # return grad_output * scale, grad_output * x, grad_output * -1, None


class Signer(torch.autograd.Function):
    """Haven't been tested yet!"""
    @staticmethod
    def forward(ctx, x, inplace):
        if inplace:
            ctx.mark_dirty(x)
            x.sign_()
            return x
        else:
            return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def affine_quant(x, scale, zero_point, inplace):
    return AffineQuantize.apply(x, scale, zero_point, inplace)


def signer(x, inplace):
    return Signer.apply(x, inplace)


def update_with_momentum(x, new_x, momentum):
    x.mul_(momentum).add_((1 - momentum) * new_x)

