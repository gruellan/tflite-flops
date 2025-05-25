# -*- coding: utf-8 -*-
"""calculate flops of tflite model. only conv and depthwise_conv considered
depends on https://pypi.org/project/tflite/
reference code
https://github.com/jackwish/tflite/blob/master/tests/test_mobilenet.py
"""

import sys
import tflite


def calc_flops(path):
    with open(path, 'rb') as f:
        buf = f.read()
        model = tflite.Model.GetRootAsModel(buf, 0)

    graph = model.Subgraphs(0)

    # print funcs
    _dict_builtin_op_code_to_name = {v: k for k, v in  tflite.BuiltinOperator.__dict__.items() if type(v) == int}
    def print_header():
        print("%-18s | %-15s | FLOPS" % ("OP_NAME", "OUTPUT SHAPE"))
        print(f"{'-'*45}")
    def print_flops(op_code_builtin, flops, out_shape):
        print("%-18s | %-15s | %d" % (
            _dict_builtin_op_code_to_name[op_code_builtin],
            out_shape,
            flops
        ))
    def print_none(op_code_builtin, out_shape):
        print("%-18s | %-15s | <IGNORED>" % (_dict_builtin_op_code_to_name[op_code_builtin], out_shape))
    def print_footer(total_flops):
        print(f"{'-'*45}")
        print("Total: %.1f M FLOPS" % (total_flops / 1.0e6))

    total_flops = 0.0
    print_header()
    for i in range(graph.OperatorsLength()):
        op = graph.Operators(i)
        op_code = model.OperatorCodes(op.OpcodeIndex())
        op_code_builtin = op_code.BuiltinCode()

        flops = 0.0
        if op_code_builtin == tflite.BuiltinOperator.CONV_2D:
            filter_shape = graph.Tensors(op.Inputs(1)).ShapeAsNumpy()
            C_out, K_h, K_w, C_in = filter_shape

            out_shape = graph.Tensors(op.Outputs(0)).ShapeAsNumpy()
            _, H_out, W_out, _ = out_shape

            flops = 2 * H_out * W_out * C_out * K_h * K_w * C_in
            print_flops(op_code_builtin, flops, out_shape)

        elif op_code_builtin == tflite.BuiltinOperator.DEPTHWISE_CONV_2D:
            filter_shape = graph.Tensors( op.Inputs(1) ).ShapeAsNumpy()
            _, K_h, K_w, C_in = filter_shape

            out_shape = graph.Tensors( op.Outputs(0) ).ShapeAsNumpy()
            _, H_out, W_out, _ = out_shape

            flops = 2 * H_out * W_out * C_in * K_h * K_w
            print_flops(op_code_builtin, flops, out_shape)

        elif op_code_builtin in [tflite.BuiltinOperator.MAX_POOL_2D, tflite.BuiltinOperator.AVERAGE_POOL_2D]:
            out_shape = graph.Tensors(op.Outputs(0)).ShapeAsNumpy()
            _, H_out, W_out, C_out = out_shape

            opt = tflite.Pool2DOptions()
            opt.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
            K_h = opt.FilterHeight()
            K_w = opt.FilterWidth()
            if op_code_builtin == tflite.BuiltinOperator.MAX_POOL_2D:
                pool_ops = K_h * K_w - 1  # comparison
            else:
                pool_ops = K_h * K_w  # add + average

            flops = pool_ops * H_out * W_out * C_out
            print_flops(op_code_builtin, flops, out_shape)

        elif op_code_builtin == tflite.BuiltinOperator.FULLY_CONNECTED:
            in_shape = graph.Tensors(op.Inputs(0)).ShapeAsNumpy()
            weight_shape = graph.Tensors(op.Inputs(1)).ShapeAsNumpy()
            out_shape = graph.Tensors(op.Outputs(1)).ShapeAsNumpy()
            batch_size, _ = in_shape
            o, i = weight_shape

            flops = 2 * batch_size * i * o
            print_flops(op_code_builtin, flops, out_shape)

        else:
            out_shape = graph.Tensors(op.Outputs(0)).ShapeAsNumpy()
            print_none(op_code_builtin, out_shape)

        total_flops += flops
    print_footer(total_flops)

if __name__ == "__main__":
  path = sys.argv[1]
  calc_flops(path)

