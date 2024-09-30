import torch
import torch.nn as nn
from .base_adapter import BaseAdapter


from ..utils.bdn import BalancedDomainNormalization2d
from ..utils.utils import set_named_submodule, get_named_submodule


class UNITTA_BDN(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        self.num_classes = cfg.CORRUPTION.NUM_CLASS
        self.num_domains = len(cfg.CORRUPTION.TYPE)
        self.eta = cfg.ADAPTER.UNITTA_BDN.ETA
        self.mode = cfg.ADAPTER.UNITTA_BDN.MODE
        self.layer = cfg.ADAPTER.UNITTA_BDN.LAYER
        self.filt = cfg.ADAPTER.UNITTA_BDN.FILT
        self.mode_domain_prediction = cfg.ADAPTER.UNITTA_BDN.MODE_DOMAIN_PREDICTION
        self.prune = cfg.ADAPTER.UNITTA_BDN.PRUNE
        self.max_domains = cfg.ADAPTER.UNITTA_BDN.MAX_DOMAINS
        self.dist_metric = cfg.ADAPTER.UNITTA_BDN.DIST_METRIC

        super().__init__(cfg, model, optimizer)

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        batch_data, domain = batch_data

        with torch.no_grad():
            model.eval()
            self.set_bn_label_domain(self.model, None, None)

            logit = model(batch_data)

        return self.update_model(model, optimizer, batch_data, logit, domain)

    def update_model(self, model, optimizer, batch_data, logit, domain):
        if self.mode == "upper":
            p_l = logit.argmax(dim=1)

            model.train()

            self.set_bn_label_domain(model, p_l, None)

            model(batch_data)

            self.set_bn_label_domain(model, p_l, domain)

            logit3 = model(batch_data)

            return logit3

        elif self.mode in ["dynamic", "static"]:
            p_l = logit.argmax(dim=1)

            model.train()

            bn_name = self.layer

            self.set_bn_label_domain(model, p_l, None)

            logit2 = model(batch_data)
            probs2 = logit2.softmax(dim=-1)
            max_probs2, _ = torch.max(probs2, dim=-1)

            domain_pred, do_expand_domain, do_prune_domain, num_domains = (
                self.get_domain_pred_expand(model, bn_name)
            )

            self.set_bn_label_domain(
                model,
                p_l,
                domain_pred,
                do_expand_domain,
                do_prune_domain,
            )

            logit3 = model(batch_data)
            probs3 = logit3.softmax(dim=-1)
            if self.filt:
                max_probs3, _ = torch.max(probs3, dim=-1)
                mask = max_probs2.ge(max_probs3)
                probs3[mask, :] = probs2[mask, :]

            return probs3

        else:
            Exception("mode not supported")

    @staticmethod
    def set_bn_label_domain(
        model,
        label,
        domain=None,
        do_expand_domain=False,
        do_prune_domain=False,
    ):
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, BalancedDomainNormalization2d):
                sub_module.label = label
                sub_module.domain = domain
                sub_module.do_expand_domain = do_expand_domain
                sub_module.do_prune_domain = do_prune_domain

        return

    @staticmethod
    def get_domain_pred_expand(model, bn_name):
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, BalancedDomainNormalization2d):
                if name == bn_name:
                    return (
                        sub_module.domain_pred,
                        sub_module.do_expand_domain,
                        sub_module.do_prune_domain,
                        sub_module.num_domains,
                    )

        return

    def configure_model(self, model: nn.Module):
        model.requires_grad_(False)
        normlayer_names = []

        for name, sub_module in model.named_modules():
            if isinstance(sub_module, nn.BatchNorm2d) or isinstance(
                sub_module, nn.BatchNorm1d
            ):
                normlayer_names.append(name)

        for name in normlayer_names:
            print(f"configure_model: {name}")
            bn_layer = get_named_submodule(model, name)

            momentum_bn = BalancedDomainNormalization2d(
                bn_layer=bn_layer,
                num_classes=self.num_classes,
                num_domains=self.num_domains,
                momentum_a=self.eta,
                gamma=0.0,
                dynamic=True if self.mode == "dynamic" else False,
                self_training=False,
                pruning=self.prune,
                max_domains=self.max_domains,
                dist_metric=self.dist_metric,
            )

            if self.mode in ["dynamic", "static"]:
                if name == self.layer:
                    print(f"configure_domain_prediction: {name}")
                    momentum_bn.do_predict_domain = True
                    momentum_bn.mode = self.mode_domain_prediction

            set_named_submodule(model, name, momentum_bn)

        return model
