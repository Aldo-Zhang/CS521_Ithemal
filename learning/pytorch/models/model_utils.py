import torch
from typing import Any, Dict

def dump_shared_params(module):
    # type: (torch.nn.Module) -> Dict[str, Any]
    return {
        name: param.data.share_memory_().untyped_storage()._share_filename_cpu_()
        for (name, param) in module.named_parameters()
    }

def load_shared_params(module, params):
    # type: (torch.nn.Module, Dict[str, Any]) -> None

    for (name, param) in module.named_parameters():
        from torch import UntypedStorage
        storage = UntypedStorage._new_shared_filename_cpu(*params[name])
        param.data = torch.tensor([], dtype=param.dtype).set_(storage).view(param.data.shape)
