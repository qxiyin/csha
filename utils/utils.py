from copy import deepcopy

import torch


def getTrainableParameters(model) -> list:
    trainableParam = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainableParam.append(name)
    return trainableParam


def getClassifierParameters(model) -> list:
    classifierParam = []
    for name, param in model.named_parameters():
        if param.requires_grad and "linear" in name:
            classifierParam.append(name)
    return classifierParam


def getFloatSubModules(Delta) -> list:
    param_float = []
    for param in Delta:
        if not "FloatTensor" in Delta[param].type():
            continue
        param_float.append(param)
    return param_float



def getNetMeta(Delta) -> (dict, dict):
    shapes = dict(((k, v.shape) for (k, v) in Delta.items()))
    sizes = dict(((k, v.numel()) for (k, v) in Delta.items()))
    return shapes, sizes


def vec2net(vec: torch.Tensor, net) -> None:
    param_float = getFloatSubModules(net)
    shapes, sizes = getNetMeta(net)
    partition = list(sizes[param] for param in param_float)
    flattenComponents = dict(zip(param_float, torch.split(vec, partition)))
    components = dict(((k, v.reshape(shapes[k])) for (k, v) in flattenComponents.items()))
    net.update(components)
    return net


def net2vec(net) -> (torch.Tensor):
    param_float = getFloatSubModules(net)

    components = []
    for param in param_float:
        components.append(net[param])
    vec = torch.cat([component.flatten() for component in components])
    return vec


def applyWeight2StateDicts(deltas, weight):
    Delta = deepcopy(deltas[0])
    param_float = getFloatSubModules(Delta)

    for param in param_float:
        Delta[param] *= 0
        for i in range(len(deltas)):
            Delta[param] += deltas[i][param] * weight[i].item()

    return Delta


def stackStateDicts(deltas):
    stacked = deepcopy(deltas[0])
    for param in stacked:
        stacked[param] = None
    for param in stacked:
        param_stack = torch.stack([delta[param] for delta in deltas], -1)
        shaped = param_stack.view(-1, len(deltas))
        stacked[param] = shaped
    return stacked

