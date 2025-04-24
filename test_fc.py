import paddle
import paddle.nn as nn
import json
import numpy as np
import os
import re
import math

try:
    from paddle_xpu.layers.nn import Linear
except ImportError:
    from paddle.nn import Linear
#from paddle.nn import Linear

import json

log_file_path = 'workerlog.0.fc_fa_mean_max'


pattern = re.compile(r".*?cases(.*)")
mean_var = r"\[mean\](-?\d+\.\d+), \[max\](\d+\.\d+)"


# 初始化列表
list_a = []
list_b = []
list_c = []
list_d = []
alpha_beta = []
a_mean = []
b_mean = []
d_mean = []
a_max = []
b_max = []
d_max = []
n = 0
try:
    with open(log_file_path, 'r') as file:
        log_lines = file.readlines()

    for i in range(len(log_lines)):
        print("i", i)
        line = log_lines[i]
        if "kXPU3" not in log_lines[i]:
            continue
        if "float16" not in log_lines[i]:
            continue
        if "fc_fusion" in log_lines[i]:
            run_mode = log_lines[i + 3]
            if "desc.run_mode" in run_mode:
                mean_match = re.search(mean_var, run_mode)
                if mean_match:
                    if float(mean_match.group(1)) == 1:
                        continue
                else:
                    print('------------------error----------------')
        else:
            continue

        a_line = log_lines[i + 1]
        match = re.search(mean_var, a_line)
        if match:
            a_mean.append(float(match.group(1)))
            a_max.append(float(match.group(2)))
        b_line = log_lines[i + 2]
        match = re.search(mean_var, b_line)
        if match:
            b_mean.append(float(match.group(1)))
            b_max.append(float(match.group(2)))
        #d_line = log_lines[i + 6]
        #match = re.search(mean_var, d_line)
        #if match:
        #    d_mean.append(float(match.group(1)))
        #    d_max.append(float(match.group(2)))
       
        if "fc_fusion" in log_lines[i]:        
            print(i)
            #match = pattern.search(line)
            #if 'cases' not in line:    
            #    # 将字符串转换为字典
            #    entry = json.loads(line)
            #else:
            #    if match:
            #        json_part = match.group(1)
            #        entry = json.loads(json_part)
            entry = json.loads(line)
            # 检查是否为"fc_fusion"操作
            if entry.get("op") == "fc_fusion":
                # 提取参数
                params = entry.get("params", {})
                desc = entry.get("desc", {})
                
                # 分别提取a, b, c, d的rows和cols，并存储在各自的列表中
                if 'a' in params:
                    if params['a']['trans']:
                        list_a.append([params['a']["cols"], params['a']["rows"]])
                    else:
                        list_a.append([params['a']["rows"], params['a']["cols"]])
                if 'b' in params:
                    if params['b']['trans']:    
                        list_b.append([params['b']["cols"], params['b']["rows"]])
                    else:
                        list_b.append([params['b']["rows"], params['b']["cols"]])
                if 'c' in params:
                    if params['c']['trans']:
                        list_c.append([params['c']["cols"], params['c']["rows"]])
                    else:
                        list_c.append([params['c']["rows"], params['c']["cols"]])
                if 'd' in params:
                    if params['d']['trans']:
                        list_d.append([params['d']["cols"], params['d']["rows"]])
                    else:
                        list_d.append([params['d']["rows"], params['d']["cols"]])
                
                # 提取alpha和beta，转换为float，并存储
                alpha = float(desc.get("alpha", 0))
                beta = float(desc.get("beta", 0))
                alpha_beta.append([alpha, beta])
except FileNotFoundError:
    print(f"Error: The file {log_file_path} does not exist.")
except IOError as e:
    print(f"Error: An I/O error occurred while reading {log_file_path}: {str(e)}")

def generate_true_random_number(num_bytes):
    """ 生成真随机数，从 /dev/urandom 读取指定数量的字节 """
    # 从 /dev/urandom 读取 num_bytes 个字节
    random_bytes = os.urandom(num_bytes)
    # 将字节转换为整数
    #print(int.from_bytes(random_bytes, 'big'))
    return int.from_bytes(random_bytes, 'big')

# inf, nan
def get_rounded_num(x, round_up=True):
    if math.isinf(x) or math.isnan(x):
        msg = f"warning, x is inf or nan"
        print(msg, x)
        return x
    if abs(x) <= 1e-10:
        return 0
    
    abs_x = abs(x)
    log_x = math.log10(abs_x)
    round_log_x = math.floor(log_x) if round_up ^ (x > 0) else math.ceil(log_x)
    
    result = 10**round_log_x
    return result if x >= 0 else -result

unique_string = []
u_list_a = []
u_list_b = []
u_a_m = []
u_a_v = []
u_b_m = []
u_b_v = []

for i in range(len(a_mean)):
    a_m = get_rounded_num(a_mean[i])
    b_m = get_rounded_num(b_mean[i])
    a_v = get_rounded_num(a_max[i])
    b_v = get_rounded_num(b_max[i])
    new_string = str(list_a[i]) + str(list_b[i]) + str(a_m) + str(b_m) + str(a_v) + str(b_v)
    if new_string not in unique_string:
        unique_string.append(new_string)
        u_list_a.append(list_a[i])
        u_list_b.append(list_b[i])
        u_a_m.append(a_m)
        u_b_m.append(b_m)
        u_a_v.append(a_v)
        u_b_v.append(b_v)

print("-------------------------------------------", len(u_list_a))

list_a = u_list_a
list_b = u_list_b
a_mean = u_a_m
b_mean = u_b_m
a_max = u_a_v
b_max = u_b_v


### 输出各个列表
#print("List A:", list_a)
#print("List B:", list_b)
#print("List C:", list_c)
#print("List D:", list_d)
#print("Alpha and Beta:", alpha_beta)
#print("a_mean", a_mean)
#print("b_mean", b_mean)
#print("d_mean", d_mean)
#print("a_max", a_max)
#print("b_max", b_max)
#print("d_max", d_max)
#
#print("-------------------------------------------", len(a_mean))
#
def generate_random_array(mean, max_value, shape, seed=None):
    if seed is not None:
        np.random.seed(seed)
    # 首先生成标准正态分布的随机数组
    random_array = np.random.randn(*shape).astype(np.float32)
    # 计算当前随机数组的最大值
    current_max = random_array.max()
    # 计算缩放因子，使得新的最大值为给定的max_value
    scale_factor = max_value / current_max
    # 对数组进行缩放
    random_array *= scale_factor
    # 计算当前数组的均值
    current_mean = random_array.mean()
    # 计算偏移量，使得新的均值为给定的mean
    shift_value = mean - current_mean
    # 对数组进行偏移
    random_array += shift_value
    return random_array

#CREATE_DATA = True
#DEVICE = 'XPU'
##DEVICE = 'GPU'
#
##
#dtype = paddle.bfloat16
#paddle.set_default_dtype(dtype)
#
#if DEVICE == 'XPU' and CREATE_DATA:
#    for i in range(len(list_a)):
#        print(i)
#    #for i in range(1):
#        paddle.seed(int(generate_true_random_number(4)))
#        #x = np.random.uniform(-1, 1, list_a[i]).astype("float32")
#        x = generate_random_array(a_mean[i], a_max[i], list_a[i], int(generate_true_random_number(4)))
#        paddle.seed(int(generate_true_random_number(5)))
#        #w = np.random.uniform(-1, 1, list_b[i]).astype("float32")
#        w = generate_random_array(b_mean[i], b_max[i], list_b[i], int(generate_true_random_number(3)))
#        paddle.seed(int(generate_true_random_number(6)))
#        bias = np.random.uniform(-1, 1, [list_b[i][1]]).astype("float32")
#        paddle.seed(int(generate_true_random_number(5)))
#        out_grad = np.random.uniform(-1, 1, [list_a[i][0], list_b[i][1]]).astype("float32")
#        #out_grad = generate_random_array(d_mean[i], d_max[i], [list_a[i][0], list_b[i][1]], int(generate_true_random_number(3)))
#        x = paddle.to_tensor(x, stop_gradient=False).cast(dtype)
#        w = paddle.to_tensor(w, stop_gradient=False).cast(dtype)
#        bias = paddle.to_tensor(bias, stop_gradient=False).cast(dtype)
#        out_grad = paddle.to_tensor(out_grad, stop_gradient=True)
#        paddle.save([x, w, bias, out_grad], 'FC_INPUT/linear_' + str(i))
#
#for i in range(len(list_a)):
##for i in range(1):
#    print(i)
#    x, w, bias, out_grad = paddle.load('FC_INPUT/linear_' + str(i))
#    linear = Linear(w.shape[0], w.shape[1], bias_attr=True)
#    linear.weight.set_value(w)
#    linear.bias.set_value(bias)
#    linear.train()
#    
#    out = linear(x)
#    #paddle.save([True, [out]], 'XPU/output/linear_' + str(i))
#    out = paddle.cast(out, "float32")
#    paddle.autograd.backward(tensors=[out], grad_tensors=[out_grad])
#    paddle.save([True, [linear.weight.grad, linear.bias.grad]], 'XPU/output_backward/linear_' + str(i))
