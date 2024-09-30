import torch
import torch.nn as nn
from .base_adapter import BaseAdapter
from ..utils.bn_layers import (
    BalancedRobustBN2dV5,
    BalancedRobustBN2dEMA,
    BalancedRobustBN1dV5,
)
from ..utils.utils import set_named_submodule, get_named_submodule
from ..utils.custom_transforms import get_tta_transforms
from copy import deepcopy


def test_para(model):
    # 检查每个参数
    for name, param in model.named_parameters():
        if "bn" in name:
            print(f"Parameter of {name}: {param[0]}")
            break

    #    break
    # pass
    # if param.requires_grad:
    #    print(f"Parameter: {name}")
    # print(f"  requires_grad: {param.requires_grad}")
    # print(f"  Gradient available: {'Yes' if param.grad is not None else 'No'}")
    # print(f"  Gradient norm: {param.grad.norm().item() if param.grad is not None else 'No gradients'}")
    # print()
    #    break


class TRIBE_BBN(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        super().__init__(cfg, model, optimizer)

        self.teacher_model = deepcopy(self.model)
        for (name1, param1), (name2, param2) in zip(
            self.model.named_parameters(), self.teacher_model.named_parameters()
        ):
            if "bn" in name1:
                set_named_submodule(self.teacher_model, name2, param1)

        self.anchor_model = deepcopy(self.model)
        self.transform = get_tta_transforms(cfg)

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        with torch.no_grad():
            self.teacher_model.eval()
            ema_out = self.teacher_model(batch_data)

        self.update_model(model, optimizer, batch_data, ema_out)

        return ema_out

    def update_model(self, model, optimizer, batch_data, logit):
        p_l = logit.argmax(dim=1)

        self.teacher_model.train()
        # model.train()

        # strong_sup_aug = self.transform(batch_data)

        # teacher
        self.set_bn_label(self.teacher_model, p_l)
        self.teacher_model(batch_data)

        # self.set_bn_label(model, p_l)
        # stu_sup_out = model(strong_sup_aug)

        # entropy = self.self_softmax_entropy(ema_sup_out)
        # entropy_mask = (entropy < self.cfg.ADAPTER.TRIBE.H0 * math.log(self.cfg.CORRUPTION.NUM_CLASS))
        # l_sup = torch.nn.functional.cross_entropy(stu_sup_out, ema_sup_out.argmax(dim=-1), reduction='none')[entropy_mask].mean()

        ## regularization

        # with torch.no_grad():
        #    self.set_bn_label(self.anchor_model, p_l)
        #    source_anchor = self.anchor_model(batch_data).detach()
        # l_reg = self.cfg.ADAPTER.TRIBE.LAMBDA * torch.nn.functional.mse_loss(ema_sup_out, source_anchor, reduction='none')[entropy_mask].mean()

        # l = (l_sup + l_reg)
        # l *= 0
        # l = l_sup
        # l = l_reg

        # optimizer.zero_grad()
        # l.backward()
        # optimizer.step()

        return

    @staticmethod
    def set_bn_label(model, label=None):
        for name, sub_module in model.named_modules():
            if (
                isinstance(sub_module, BalancedRobustBN1dV5)
                or isinstance(sub_module, BalancedRobustBN2dV5)
                or isinstance(sub_module, BalancedRobustBN2dEMA)
            ):
                sub_module.label = label
        return

    @staticmethod
    def self_softmax_entropy(x):
        return -(x.softmax(dim=-1) * x.log_softmax(dim=-1)).sum(dim=-1)

    @staticmethod
    def update_ema_variables(ema_model, model, nu):
        print("update_ema_variables")
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
        return ema_model

    def configure_model(self, model: nn.Module):
        model.requires_grad_(False)
        normlayer_names = []

        for name, sub_module in model.named_modules():
            if isinstance(sub_module, nn.BatchNorm2d) or isinstance(
                sub_module, nn.BatchNorm1d
            ):
                normlayer_names.append(name)

        for name in normlayer_names:
            bn_layer = get_named_submodule(model, name)
            if isinstance(bn_layer, nn.BatchNorm2d):
                NewBN = BalancedRobustBN2dV5
                # NewBN = BalancedRobustBN2dEMA
                # NewBN = RobustBN2d
            elif isinstance(bn_layer, nn.BatchNorm1d):
                NewBN = BalancedRobustBN1dV5
                # NewBN = RobustBN1d
            else:
                raise RuntimeError()

            momentum_bn = NewBN(
                bn_layer,
                self.cfg.CORRUPTION.NUM_CLASS,
                self.cfg.ADAPTER.TRIBE.ETA,
                self.cfg.ADAPTER.TRIBE.GAMMA,
            )

            # momentum_bn = NewBN(bn_layer, 0.05)

            momentum_bn.requires_grad_(True)
            set_named_submodule(model, name, momentum_bn)
        return model
