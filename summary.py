import numpy as np
import torch
import torch.nn as nn


def summary(model, input_size, batch_size=-1, device="cuda", x=None):
    def register_hook(module):
        def hook(module, input, output):
            class_name = module._get_name()
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = {}
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        hooks.append(module.register_forward_hook(hook))

    def get_children(module):
        if list(module.children()):
            return {
                module._get_name(): [get_children(child) for child in module.children()]
            }
        else:
            return module._get_name()

    def get_value(dictionary, prefix):
        for k in dictionary:
            if k.startswith(prefix):
                return dictionary.pop(k)
        return {}

    # print(get_children(model))
    def show(children, blank="|_", seq=0):
        if isinstance(children, dict):
            for parent, child in children.items():
                show(parent, blank, seq)
                show(child, blank="| " + blank)
        elif isinstance(children, list):
            for seq, child in enumerate(children):
                show(child, blank, seq)
        else:
            value = get_value(summary, children)
            line_new = "{:<25} {:>25} {:>25} {:>15}".format(
                f"{blank[2:]}({seq}) {children}",
                str(value.get("input_shape", [])),
                str(value.get("output_shape", [])),
                "{0:,}".format(value.get("nb_params", 0)),
            )

            print(line_new)

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    if not x:
        x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = {}
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("-" * 93)
    line_new = "{:>25} {:>25} {:>25} {:>15}".format(
        "Layer (type)", "Input Shape", "Output Shape", "Param #"
    )
    print(line_new)
    print("=" * 93)

    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer] and summary[layer]["trainable"]:
            trainable_params += summary[layer]["nb_params"]

    show(get_children(model))
    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4.0 / (1024**2.0))
    total_output_size = abs(
        2.0 * total_output * 4.0 / (1024**2.0)
    )  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4.0 / (1024**2.0))
    total_size = total_params_size + total_output_size + total_input_size

    print("=" * 93)
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("-" * 93)
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("-" * 93)
    # return summary
