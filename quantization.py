# Static Quantization
from inspect import Parameter
from albumentations.augmentations.functional import scale
import torch
from torchvision import models


r = models.quantization.resnet18()
torch.save(r.state_dict(), 'resnet18_int8.pth')


class ResNet_int8(torch.nn.Module):
    def __init__(self) -> None:
        super(ResNet_int8, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = models.resnet18()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor):
        out = self.quant(x)
        out = self.model(out)
        return self.dequant(out)


# define a floating point model where some layers could be statically quantized
class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.relu = torch.nn.ReLU()
        self.relu_add = torch.nn.quantized.FloatFunctional()
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        idenitfy = x
        out = self.conv(x)
        print(out.dtype, idenitfy.dtype)
        out = torch.add(out, idenitfy)
        #out = self.relu_add.add_relu(idenitfy, out)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(out)
        return x

# create a model instance
model_fp32 = M()

# model must be set to eval mode for static quantization logic to work
model_fp32.eval()

# attach a global qconfig, which contains information about what kind
# of observers to attach. Use 'fbgemm' for server inference and
# 'qnnpack' for mobile inference. Other quantization configurations such
# as selecting symmetric or assymetric quantization and MinMax or L2Norm
# calibration techniques can be specified here.
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Fuse the activations to preceding layers, where applicable.
# This needs to be done manually depending on the model architecture.
# Common fusions include `conv + relu` and `conv + batchnorm + relu`
quantization_list = []
for name, mod in model_fp32.named_modules():
    if isinstance(mod, torch.nn.Conv2d):
        quantization_list.append(name)
    # elif isinstance(mod, torch.nn.BatchNorm2d):
    #     quantization_list.append(name)
    elif isinstance(mod, torch.nn.ReLU):
        quantization_list.append(name)
print(quantization_list)

# model_fp32_fused = torch.quantization.fuse_modules(model_fp32, 
#     [['model.conv1', 'model.relu'], ['model.layer1.0.conv1']])

# Prepare the model for static quantization. This inserts observers in
# the model that will observe activation tensors during calibration.
model_fp32_prepared = torch.quantization.prepare(model_fp32)

# calibrate the prepared model to determine quantization parameters for activations
# in a real world setting, the calibration would be done with a representative dataset
input_fp32 = torch.randn(4, 1, 32, 32)
model_fp32_prepared(input_fp32)

# Convert the observed model to a quantized model. This does several things:
# quantizes the weights, computes and stores the scale and bias value to be
# used with each activation tensor, and replaces key operators with quantized
# implementations.
model_int8 = torch.quantization.convert(model_fp32_prepared)

# run the model, relevant calculations will happen in int8
res = model_int8(input_fp32)

## NNI
from nni.compression.torch import *
import torch.nn.functional as F
import copy

class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 5, 5, 1)
        self.conv2 = torch.nn.Conv2d(5, 2, 5, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return F.log_softmax(x, dim=1)
model = TorchModel()

dummy_input = torch.randn(1,1,9,9)
model(dummy_input)

model_quant = copy.deepcopy(model)
model_quant(dummy_input)


def conv_quant(conv):
    weight = conv.weight
    bias = conv.bias


weight = copy.deepcopy(model.conv2.weight)
new_scale = weight.abs().max() / 127
scale_ = max(0, new_scale)
orig_type = weight.type()
weight = weight.div(scale_).type(torch.int8)
model.conv2.weight = torch.nn.Parameter(weight, requires_grad=False)
model.conv2.weight.dtype

model_quant(dummy_input)
model_quant.conv1.weight.dtype
model(dummy_input)
model.conv1.weight.dtype

bias = copy.deepcopy(model.conv1.bias)
new_scale = bias.abs().max() / 127
scale_ = max(0, new_scale)
orig_type = bias.type()
bias = bias.div(scale_).type(torch.int8)
model.conv1.bias = torch.nn.Parameter(bias, requires_grad=False)
model.conv1.bias.dtype
model.conv1(dummy_input.type(torch.int8))




print('done.')