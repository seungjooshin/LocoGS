import torch
import numpy as np

class HalfSTEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.half().float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
def half_ste(input):
    return HalfSTEFunction.apply(input)

def log_transform(data):
    # Implementation from Self-Organizing-Gaussians:
    # https://github.com/fraunhoferhhi/Self-Organizing-Gaussians/blob/5f2b360abf496af560ded8ddc13157a601dade32/compression/compression_exp.py#L70
    positive = data > 0
    negative = data < 0

    data[positive] = np.log1p(data[positive])
    data[negative] = -np.log1p(-data[negative])

    return data

def inverse_log_transform(data):
    # Implementation from Self-Organizing-Gaussians:
    # https://github.com/fraunhoferhhi/Self-Organizing-Gaussians/blob/5f2b360abf496af560ded8ddc13157a601dade32/compression/compression_exp.py#L82
    positive = data > 0
    negative = data < 0

    data[positive] = np.expm1(data[positive])
    data[negative] = -np.expm1(-data[negative])

    return data

def distributal_clip(data, bit=8):
    d = 3 + 3 * (bit - 1) / 15
    mean, std = data.mean(), data.std()
    return data.clip(mean - d * std, mean + d * std)

def quantize(data, bit=8, log=False, clip=True):
    if clip:
        data = distributal_clip(data, bit)
    if log:
        data = log_transform(data)

    data_max, data_min = data.max(), data.min()
    data_scale = (2**bit - 1) / (data_max - data_min)
    data_q = np.clip(np.round((data - data_min) * data_scale), 0, 2**bit - 1).astype(np.uint8)

    return data_q, data_scale, data_min

def dequantize(data_quant, data_scale, data_min, log=False):
    data_q = (data_quant.astype(np.float32) / data_scale) + data_min
    if log:
        data_q = inverse_log_transform(data_q)

    return data_q
