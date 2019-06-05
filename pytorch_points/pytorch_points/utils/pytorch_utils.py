import torch
import numpy as np
import os
from collections import OrderedDict


def save_network(net, directory, network_label, epoch_label=None, **kwargs):
    save_filename = "_".join((network_label, str(epoch_label))) + ".pth"
    save_path = os.path.join(directory, save_filename)
    merge_states = OrderedDict()
    merge_states['states'] = net.cpu().state_dict()
    for k in kwargs:
        merge_states[k] = kwargs[k]
    torch.save(merge_states, save_path)
    net = net.cuda()


def load_network(net, path):
    """
    load network parameters whose name exists in the pth file.
    return:
        INT trained step
    """
    if path[-3:] == "pth":
        loaded_state = torch.load(path)
    else:
        loaded_state = np.load(path).item()
    loaded_param_names = set(loaded_state["states"].keys())
    network = net.module if isinstance(
        net, torch.nn.DataParallel) else net

    # allow loaded states to contain keys that don't exist in current model
    # by trimming these keys;
    own_state = network.state_dict()
    extra = loaded_param_names - set(own_state.keys())
    if len(extra) > 0:
        print('Dropping ' + str(extra) + ' from loaded states')
    for k in extra:
        del loaded_state["states"][k]

    try:
        network.load_state_dict(loaded_state["states"])
    except KeyError as e:
        print(e)
        return 0
    else:
        print('Loaded network parameters from {}'.format(path))
        if "step" in loaded_state:
            return loaded_state["step"]
        else:
            return 0


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def tolerating_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = [x for x in filter(lambda x: x is not None, batch)]
    return torch.utils.data.dataloader.default_collate(batch)
