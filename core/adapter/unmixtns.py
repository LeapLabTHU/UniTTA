import torch
import torch.nn as nn
from .base_adapter import BaseAdapter

# import torch.nn.functional as F
from ..utils.unmixtns import replace_bn_layers


class UNMIXTNS(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        super().__init__(cfg, model, optimizer)
        return

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        outputs = model(batch_data)
        return outputs

    def configure_model(self, model: nn.Module):
        model.requires_grad_(False)
        # _C.UNMIXTNS.NUM_COMPONENTS = 16
        # _C.UNMIXTNS.BATCH_SIZE_MAX = 64

        print("Replacing BatchNorm layers with UnMix-TNS layers")
        replace_bn_layers(
            model,
            self.cfg.ADAPTER.UNMIXTNS.NUM_COMPONENTS,
            self.cfg.TEST.BATCH_SIZE,
            self.cfg.ADAPTER.UNMIXTNS.BATCH_SIZE_MAX,
        )

        # for module in model.modules():
        #    if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
        #        # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        #        # TENT: force use of batch stats in train and eval modes: https://github.com/DequanWang/tent/blob/master/tent.py
        #        module.track_running_stats = False
        #        module.running_mean = None
        #        module.running_var = None

        return model
