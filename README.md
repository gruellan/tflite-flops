# tflite-flops
Roughly calculate FLOPs (floating-point operations) of a TFLite format model.
Supported layers:
- Conv2D
- DepthwiseConv2D
- MaxPool2D
- AveragePool2D
- Dense

### Install
```
pip3 install git+https://github.com/gruellan/tflite-flops
```

### Usage
```
python3 -m tflite_flops example.tflite
```

### Example
```
wget https://storage.googleapis.com/tf_model_garden/vision/qat/mobilenetv2_ssd_coco/model_int8_qat.tflite
python3 -m tflite_flops ./model_int8_qat.tflite
```
below lines printed
```
OP_NAME                 | OUTPUT SHAPE         | FLOPS
---------------------------------------------
QUANTIZE                | [  1 256 256   3]    | <IGNORED>
CONV_2D                 | [  1 128 128  32]    | 28311552
DEPTHWISE_CONV_2D       | [  1 128 128  32]    | 9437184
CONV_2D                 | [  1 128 128  16]    | 16777216
.
.
.
CONCATENATION           | [    1 12276     4]  | <IGNORED>
DEQUANTIZE              | [    1 12276     4]  | <IGNORED>
CUSTOM                  | []                   | <IGNORED>
---------------------------------------------
Total: 1485.7 M FLOPS
```

### How is it calculated?

In the case of Conv2D layer
```
Multiply-Accumulate (MAC) = output_h * output_w * output_c * kernel_h * kernel_w * input_c 
                         (= output_h * output_w * weight_size)
Floating-point operations (FLOPs) = 2 * MAC
```
