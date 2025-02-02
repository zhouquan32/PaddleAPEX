# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import paddle.distributed as dist
import paddle
from .. import config
from ..api_info import API, get_init_params, save_init_params_and_weight, save_init_params, save_weight
import os
from paddleapex.api_tracer.Dump import dump_util


class HookOp:
    pass


cfg = config.cfg


def hijack_init(self, *args, **kwargs):
    print("args", args)
    print("kwargs", kwargs)
    self.__init__(*args, **kwargs)


def create_hook_with_info(tensor, attr_index, api):
    def grad_hook(grad):
        if grad is not None:
            single_arg = {}
            single_arg.update({"type": "paddle.Tensor"})
            single_arg.update({"dtype": str(grad.dtype.name)})
            single_arg.update({"shape": grad.shape})
            single_arg.update({"stop_gradient": grad.stop_gradient})
            api_args = api.op_name + ".grad_" + str(attr_index)
            pt_path = dump_util.dump_real_data(api_args, grad.detach().cpu(), api.rank)
            single_arg.update({"real_data_path": pt_path})

            api.dout_list.append(single_arg)
            api.output_num -= 1
            if api.output_num == 0:
                api.api_info_struct[api.op_name].update({"dout_list": api.dout_list})
    if api.mode == "real_data":
        return grad_hook
    else:
        return api.record_dout


def create_output_attr(tensor, num):
    setattr(tensor, 'id_apex', num)
    return 'id_apex', num


def hijack_call(self, *args, **kwargs):
    cls = self.__class__
    # init_params = get_init_params(self)
    cfg.prefix_op_name_ = self.prefix_op_name_ + "*"
    if self.__class__.__name__ not in cfg.Op_count:
        cfg.Op_count[self.__class__.__name__] = 1
        cfg.prefix_op_name_ += "0"
    else:
        cfg.Op_count[self.__class__.__name__] += 1
        cfg.prefix_op_name_ += str(cfg.Op_count[self.__class__.__name__] - 1)
    if cfg.dump_state:
        api_recorder = API(cfg.dump_mode)
        rank = dist.get_rank()
        api_recorder.update_APIInfo(cfg.prefix_op_name_, rank)
        api_recorder.update_real_data(args, kwargs)
        # save_weight(self.state_dict(), cfg.prefix_op_name_, rank)
        # save_init_params_and_weight(init_params, self.state_dict(), cfg.prefix_op_name_, rank)
        save_init_params_and_weight(self.apex_init_params, self.state_dict(), cfg.prefix_op_name_, rank)
        output = self.forward(*args, **kwargs)
        try:
            out_num = 0
            if isinstance(output, paddle.Tensor):
                if not output.stop_gradient:
                    output.register_hook(create_hook_with_info(output, api_recorder.output_num, api_recorder))
                    #output.register_hook(api_recorder.record_dout)
                    api_recorder.output_num = 1
                else:
                    api_recorder.record_dout(None)
            if isinstance(output, (list, tuple)):
                need_record = False
                for item in output:
                    if isinstance(item, paddle.Tensor) and not item.stop_gradient:
                        item.register_hook(create_hook_with_info(item, api_recorder.output_num, api_recorder))
                        api_recorder.output_num += 1
                        need_record = True
                if not need_record:
                    api_recorder.record_dout(None)
        except Exception as e:
            print(self.__class__.__name__, " register hook failed. Due to :", e)
            api_recorder.record_dout(None)
    else:
        output = self.forward(*args, **kwargs)
    return output


class OPTemplate:
    def __init__(self, op_name):
        self.op_name_ = op_name
        cfg.prefix_op_name_ = self.op_name_ + "*"

    def forward(self, *args, **kwargs):
        if self.op_name_ not in cfg.Op_count:
            cfg.Op_count[self.op_name_] = 1
            cfg.prefix_op_name_ += "0"
        else:
            cfg.Op_count[self.op_name_] += 1
            cfg.prefix_op_name_ += str(cfg.Op_count[self.op_name_] - 1)
        if cfg.dump_state:
            api_recorder = API(cfg.dump_mode)
            rank = dist.get_rank()
            api_recorder.update_APIInfo(cfg.prefix_op_name_, rank)
            api_recorder.update_real_data(args, kwargs)
            output = getattr(HookOp, "wrap_" + str(self.op_name_))(*args, **kwargs)
            try:
                if isinstance(output, paddle.Tensor):
                    if not output.stop_gradient:
                        #output.register_hook(api_recorder.record_dout)
                        output.register_hook(create_hook_with_info(output, api_recorder.output_num, api_recorder))
                        api_recorder.output_num = 1
                    else:
                        api_recorder.record_dout(None)
                if isinstance(output, (list, tuple)):
                    need_record = False
                    for item in output:
                        if isinstance(item, paddle.Tensor) and not item.stop_gradient:
                            need_record = True
                            #item.register_hook(api_recorder.record_dout)
                            item.register_hook(create_hook_with_info(item, api_recorder.output_num, api_recorder))
                            api_recorder.output_num += 1
                    if not need_record:
                        api_recorder.record_dout(None)
            except Exception as e:
                print(self.op_name_, " register hook failed. Due to :", e)
                api_recorder.record_dout(None)
        else:
            output = getattr(HookOp, "wrap_" + str(self.op_name_))(*args, **kwargs)
        return output

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)
