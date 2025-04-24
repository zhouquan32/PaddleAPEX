import paddle
import paddle.nn as nn
import json
import numpy as np
import os
import re

try:
    from paddle_xpu.layers.nn import Linear
except ImportError:
    from paddle.nn import Linear

import json
CREATE_DATA = False
#DEVICE = 'XPU'
DEVICE = 'GPU'

dtype = paddle.bfloat16
paddle.set_default_dtype(dtype)

for i in range(len(os.listdir('FC_INPUT/'))):
#for i in range(1):
    print(i)
    x, w, bias, out_grad = paddle.load('FC_INPUT/linear_' + str(i))
    linear = Linear(w.shape[0], w.shape[1], bias_attr=True)
    linear.weight.set_value(w)
    linear.bias.set_value(bias)
    linear.train()

    out = linear(x)
    paddle.save([True, [out]], 'GPU/output/linear_' + str(i))
    out = paddle.cast(out, "float32")
    paddle.autograd.backward(tensors=[out], grad_tensors=[out_grad])
    paddle.save([True, [linear.weight.grad, linear.bias.grad]], 'GPU/output_backward/linear_' + str(i))
