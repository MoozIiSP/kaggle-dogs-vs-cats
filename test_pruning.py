# %%
from timeit import timeit
from functools import reduce

import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from nni.compression.torch import *
# from rich import print
from torch import nn
from torchvision import models

# %%
# Test torch.sparse.Tensor (1.9ms) and torch.Tensor (4.73ms)
tensor = torch.randint(0, 2, size=(2048, 2048), dtype=torch.float)
stensor = torch.sparse.Tensor(torch.randint(0, 2, size=(2048, 2048), dtype=torch.float))

print('mul: {:.2f}ms'.format(
    1000 * timeit('torch.Tensor.mul(tensor, stensor)', number=1000, globals=globals())))
print('add: {:.2f}ms'.format(
    1000 * timeit('torch.Tensor.add(tensor, stensor)', number=1000, globals=globals())))
print('sparse_mul: {:.2f}ms'.format(
    1000 * timeit('torch.sparse.Tensor.mul(tensor, stensor)', number=1000, globals=globals())))
print('sparse_add: {:.2f}ms'.format(
    1000 * timeit('torch.sparse.Tensor.mul(tensor, stensor)', number=1000, globals=globals())))


# %%
# TODO: Porting for NNI
# TODO: Write LitPruningTraining?
#   And deploy model onto the embedded device.
net = models.resnet50(num_classes=2)
pruned_net = models.resnet50(num_classes=2)

dummy_input = torch.randn(8, 3, 224, 224)


# Inspect shape of all parameters after pruning
# Inspect a Module
# mod = pruned_net.conv1
# print(list(mod.named_parameters()))
# print(list(mod.named_buffers()))

# pruning_targets = []

# NOTE:
#   prune.random_unstructured
#   prune.l1_unstructured
#   prune.random_structured
#   prune.ln_structured
# unstrcutured for weight pruning and structured for channel pruning

# Iteration over named_parameters
for name, mod in pruned_net.named_modules():
    # if hasattr(mod, 'bias'):
    #     prune.random_unstructured(mod, name="bias", amount=0.5)
    # elif hasattr(mod, 'weight'):
    #     prune.random_unstructured(mod, name="weight", amount=0.5)
    # else:
    if isinstance(mod, torch.nn.Conv2d):
        prune.random_unstructured(mod, name="weight", amount=0.5)
    elif isinstance(mod, torch.nn.BatchNorm2d):
        prune.random_unstructured(mod, name="weight", amount=0.5)
        prune.random_unstructured(mod, name="bias", amount=0.5)
    elif isinstance(mod, torch.nn.Linear):
        prune.random_unstructured(mod, name="weight", amount=0.5)
        prune.random_unstructured(mod, name="bias", amount=0.5)


# Apply prune
for name, mod in pruned_net.named_modules():
    if isinstance(mod, torch.nn.Linear):
        prune.remove(mod, name='weight')
        prune.remove(mod, name='bias')
    elif isinstance(mod, torch.nn.BatchNorm2d):
        prune.remove(mod, name='weight')
        prune.remove(mod, name='bias')
    elif isinstance(mod, torch.nn.Conv2d):
        prune.remove(mod, name='weight')


# Count Sparsity
def count_sparsity(net):
    zero_weight = torch.tensor(0.)
    zero_bias = torch.tensor(0.)
    tot_weight = torch.tensor(0.)
    tot_bias = torch.tensor(0.)

    for name, params in net.named_parameters():
        if 'weight' in name:
            zero_weight += torch.sum(params == 0)
            tot_weight += params.nelement()
        elif 'bias' in name:
            zero_bias += torch.sum(params == 0)
            tot_bias += params.nelement()

    return zero_weight, tot_weight, zero_bias, tot_bias

# Count Parameters
# print('{:.2f} MB'.format(
#       reduce(lambda x,y:x+y, [param.nelement() for param in net.parameters()]) / 1024 / 1024))

zw, tw, zb, tb = count_sparsity(net)
print('Params: {:.2f}MB, Sparsity: {:.2f}'.format(
    (tw + tb) / 2**20, (zw + zb) / (tw + tb)
))
zw, tw, zb, tb = count_sparsity(pruned_net)
print('Params: {:.2f}MB, Sparsity: {:.2f}'.format(
    (tw + tb) / 2**20, (zw + zb) / (tw + tb)
))

print('net: {:.2f}ms'.format(
    1000 * timeit('net(dummy_input)', number=10, globals=globals())))
print('pruned: {:.2f}ms'.format(
    1000 * timeit('pruned_net(dummy_input)', number=10, globals=globals())))

# %%
net = models.vgg11(num_classes=2)
pruned_net = models.vgg11(num_classes=2)

dummy_input = torch.randn(8, 3, 224, 224)


# Inspect shape of all parameters after pruning
# Inspect a Module
# mod = pruned_net.conv1
# print(list(mod.named_parameters()))
# print(list(mod.named_buffers()))

# pruning_targets = []

# NOTE:
#   prune.random_unstructured
#   prune.l1_unstructured
#   prune.random_structured
#   prune.ln_structured
# unstrcutured for weight pruning and structured for channel pruning

# Iteration over named_parameters
for name, mod in pruned_net.named_modules():
    # if hasattr(mod, 'bias'):
    #     prune.random_unstructured(mod, name="bias", amount=0.5)
    # elif hasattr(mod, 'weight'):
    #     prune.random_unstructured(mod, name="weight", amount=0.5)
    # else:
    if isinstance(mod, torch.nn.Conv2d):
        prune.random_structured(mod, name="weight", amount=0.5, dim=0)
    elif isinstance(mod, torch.nn.BatchNorm2d):
        pass
        # print(name, mod.weight.shape, ' dim must big than 1.')
    #     prune.random_structured(mod, name="weight", amount=0.5, dim=0)
    elif isinstance(mod, torch.nn.Linear):
        prune.random_structured(mod, name="weight", amount=0.5, dim=1)


# # Apply prune
# for name, mod in pruned_net.named_modules():
#     if isinstance(mod, torch.nn.Linear):
#         prune.remove(mod, name='weight')
#     # elif isinstance(mod, torch.nn.BatchNorm2d):
#     #     prune.remove(mod, name='weight')
#     elif isinstance(mod, torch.nn.Conv2d):
#         prune.remove(mod, name='weight')


# # Remove bn with related to conv will be removed.
# def reconstructing_channel(net):
#     # Remove channel of which sum is equal to 0.
#     pruned_list = {}  # You should save the whole model not weights.
#     for name, mod in pruned_net.named_modules():
#         if isinstance(mod, torch.nn.Conv2d):
#             dim = list(range(1, len(mod.weight.shape[1:])+1))
#             ind = torch.sum(mod.weight, dim=dim) == 0
#             pruned_list[name] = ind
#             # print(name, f"{ind.nonzero().numpy()} will be removed.")
#         elif isinstance(mod, torch.nn.Linear):
#             ind = torch.sum(mod.weight, dim=0) == 0
#             pruned_list[name] = ind
#             # print(name, f"{ind.nonzero().numpy()} will be removed.")

#     last_out = torch.tensor([True]*3)
#     for name, mod in net.named_modules():
#         if isinstance(mod, torch.nn.Conv2d):
#             reverse = ~pruned_list[name]
#             mod.in_channels /= 2
#             mod.out_channels /= 2
#             print(name, mod.in_channels, mod.out_channels)
#             mod.weight = nn.Parameter(mod.weight[reverse])
#             mod.bias = nn.Parameter(mod.bias[reverse])
#             last_out = reverse
#         elif isinstance(mod, torch.nn.BatchNorm2d):
#             print(name)
#             mod.num_features /= 2
#             print(name, mod.num_features)
#             mod.weight = nn.Parameter(mod.weight[last_out])
#             mod.bias = nn.Parameter(mod.bias[last_out])
#             mod.running_mean = nn.Parameter(mod.running_mean[last_out], requires_grad=False)
#             mod.running_var = nn.Parameter(mod.running_var[last_out], requires_grad=False)
#         #     prune.random_structured(mod, name="weight", amount=0.5, dim=0)
#         elif isinstance(mod, torch.nn.Linear):
#             print(name)
#             reverse = ~pruned_list[name]
#             mod.in_features /= 2
#             mod.out_features /= 2
#             print(name, mod.in_features)
#             mod.weight = nn.Parameter(mod.weight[:, reverse])
#             # if torch.sum(reverse) > 2:
#             #     mod.bias = nn.Parameter(mod.bias[reverse])

# TODO: 通过hook来移除通道，因为复杂的网络拓扑图不一定是顺序的。
pruned_net = models.resnet50(num_classes=2)
@torch.no_grad()
def remove_hook(mod):
    def hook_fn(mod, tensors) -> None:
        last_feats = tensors.size(1)
        if hasattr(mod, 'weight'):
            if hasattr(mod, 'in_channels'):
                mask = list(mod.named_buffers())[0][1]
                inds = torch.sum(mask, dim=[1,2,3]).bool()
                print(mod.in_channels, mod.out_channels)
                print(inds.shape, inds)
            elif hasattr(mod, 'num_features'):
                print(mod.num_features)
            else:
                print(mod.in_features, mod.out_features)
                print(list(mod.named_buffers()))
    mod.register_forward_pre_hook(hook_fn)

pruned_net.apply(remove_hook)
pruned_net(torch.randn(1,3,64,64))


def pruning_model():
    # Add Hook
    def pruning_hook(mod):
        pass

    # Execute pruning

    # Remove Hook
    pass

# # Count Sparsity
# def count_sparsity(net):
#     zero_weight = torch.tensor(0.)
#     zero_bias = torch.tensor(0.)
#     tot_weight = torch.tensor(0.)
#     tot_bias = torch.tensor(0.)

#     for name, params in net.named_parameters():
#         if 'weight' in name:
#             zero_weight += torch.sum(params == 0)
#             tot_weight += params.nelement()
#         elif 'bias' in name:
#             zero_bias += torch.sum(params == 0)
#             tot_bias += params.nelement()

#     return zero_weight, tot_weight, zero_bias, tot_bias

# # Count Parameters
# # print('{:.2f} MB'.format(
# #       reduce(lambda x,y:x+y, [param.nelement() for param in net.parameters()]) / 1024 / 1024))

# zw, tw, zb, tb = count_sparsity(net)
# print('Params: {:.2f}MB, Sparsity: {:.2f}'.format(
#     (tw + tb) / 2**20, (zw + zb) / (tw + tb)
# ))
# zw, tw, zb, tb = count_sparsity(pruned_net)
# print('Params: {:.2f}MB, Sparsity: {:.2f}'.format(
#     (tw + tb) / 2**20, (zw + zb) / (tw + tb)
# ))

# # print('net: {:.2f}ms'.format(
# #     1000 * timeit('net(dummy_input)', number=10, globals=globals())))
# # print('pruned: {:.2f}ms'.format(
# #     1000 * timeit('pruned_net(dummy_input)', number=10, globals=globals())))


# # FIXME: Remove, but cannot forward
# reconstructing_channel(pruned_net)
# for _, param in pruned_net.named_parameters():
#     print(_, param.shape)

# zw, tw, zb, tb = count_sparsity(pruned_net)
# print('Params: {:.2f}MB, Sparsity: {:.2f}'.format(
#     (tw + tb) / 2**20, (zw + zb) / (tw + tb)
# ))
# print('pruned: {:.2f}ms'.format(
#     1000 * timeit('pruned_net(dummy_input)', number=10, globals=globals())))





# %%
# NNI Pruning Technology, but it will not reduce number of parameters without apply compression result.
net = models.resnet50(num_classes=2)
dummy_input = torch.randn(1,3,224,224)

print('net: {:.2f}ms'.format(
    1000 * timeit('net(dummy_input)', number=20, globals=globals())))

# config_list = [{ 'sparsity': 0.5, 'op_types': ['Conv2d'] }] 
# pruner = LevelPruner(net, config_list)
config_list = [{ 'sparsity': 0.8, 'op_types': ['BatchNorm2d'] }]
pruner = SlimPruner(net, config_list)
pruned_net = pruner.compress()
print('pruned: {:.2f}ms'.format(
    1000 * timeit('pruned_net(dummy_input)', number=20, globals=globals())))





# 剪枝之后会生成weight_mask，这个mask会不会占用空间，也不会加速推理速度😂
#   Ans: 序列化保存之后，会发现mask确实会占用空间。但是可以使剪枝永久化？怎么永久化？
# 那么剪枝的标准是什么？在不影响计算的情况下，剪枝的最小粒度应该是多大？
# prune.random_unstructured(mod, name="weight", amount=0.3)


# Group Pruning
# So far, we only looked at what is usually referred to as “local” pruning, 
# i.e. the practice of pruning tensors in a model one by one, by comparing 
# the statistics (weight magnitude, activation, gradient, etc.) of each 
# entry exclusively to the other entries in that tensor. However, a common 
# and perhaps more powerful technique is to prune the model all at once, 
# by removing (for example) the lowest 20% of connections across the whole 
# model, instead of removing the lowest 20% of connections in each layer. 
# This is likely to result in different pruning percentages per layer. 
# Let’s see how to do that using global_unstructured from torch.nn.utils.prune.


# 确定稀疏性是统计权重中的0的个数
# 那稀疏矩阵会占多大空间呢，还是说在运行时展开占用内存，存储时占用少？
# print(
#     "Sparsity in conv1.weight: {:.2f}%".format(
#         100. * float(torch.sum(model.conv1.weight == 0))
#         / float(model.conv1.weight.nelement())
#     )
# )

# custom pruning method


print('debug done.')

# %%
