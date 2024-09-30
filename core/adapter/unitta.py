# import math
import torch
import torch.nn as nn
from .base_adapter import BaseAdapter

from ..utils.bdn import BalancedDomainNormalization2d
from ..utils.utils import set_named_submodule, get_named_submodule


class UNITTA(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        self.last_features1 = None
        self.last_features2 = None
        self.last_features3 = None
        self.num_classes = cfg.CORRUPTION.NUM_CLASS
        self.eta = cfg.ADAPTER.UNITTA.ETA
        self.gamma = cfg.ADAPTER.UNITTA.GAMMA
        self.layer = cfg.ADAPTER.UNITTA.LAYER
        self.classifier_name = cfg.ADAPTER.UNITTA.CLASSIFIER
        self.mode_domain_prediction = cfg.ADAPTER.UNITTA.MODE_DOMAIN_PREDICTION
        self.filt = cfg.ADAPTER.UNITTA.FILT
        self.prune = cfg.ADAPTER.UNITTA.PRUNE
        self.max_domains = cfg.ADAPTER.UNITTA.MAX_DOMAINS
        self.dist_metric = cfg.ADAPTER.UNITTA.DIST_METRIC

        super().__init__(cfg, model, optimizer)

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, _):
        with torch.no_grad():
            model.eval()
            self.set_bn_label_domain(self.model, None, None)

            logit = self.forward_cofa(model(batch_data), flag=1)

        return self.update_model(model, batch_data, logit)

    def forward_cofa(self, feats, flag):
        probas = self.classifier(feats).softmax(dim=-1)  # [N, K]
        max_probs, _ = torch.max(probas, dim=-1)

        new_feats = torch.zeros_like(feats).to(feats.device)
        new_feats[1:] = (feats[1:] + feats[:-1]) / 2

        if flag == 1:
            if self.last_features1 is not None:
                new_feats[0] = (feats[0] + self.last_features1) / 2
            else:
                new_feats[0] = feats[0]

            self.last_features1 = feats[-1]
        elif flag == 2:
            if self.last_features2 is not None:
                new_feats[0] = (feats[0] + self.last_features2) / 2
            else:
                new_feats[0] = feats[0]

            self.last_features2 = feats[-1]
        elif flag == 3:
            if self.last_features3 is not None:
                new_feats[0] = (feats[0] + self.last_features3) / 2
            else:
                new_feats[0] = feats[0]

            self.last_features3 = feats[-1]
        else:
            raise ValueError("flag should be 1, 2 or 3")

        new_probs = self.classifier(new_feats).softmax(dim=-1)  # [N, K]
        max_new_probs, _ = torch.max(new_probs, dim=-1)

        mask = max_probs.ge(max_new_probs)
        new_probs[mask, :] = probas[mask, :]
        return new_probs

    def update_model(self, model, batch_data, logit):
        probs = logit.softmax(dim=-1)
        max_probs, _ = torch.max(probs, dim=-1)

        p_l = logit.argmax(dim=1)

        model.train()

        bn_name = self.layer
        self.set_bn_label_domain(model, p_l, None)
        logit2 = self.forward_cofa(model(batch_data), flag=2)
        probs2 = logit2.softmax(dim=-1)
        max_probs2, _ = torch.max(probs2, dim=-1)

        domain_pred, do_expand_domain, do_prune_domain = self.get_domain_pred_expand(
            model, bn_name
        )

        self.set_bn_label_domain(
            model,
            p_l,
            domain_pred,
            do_expand_domain,
            do_prune_domain,
        )

        probs3 = self.forward_cofa(model(batch_data), flag=3)

        if self.filt:
            max_probs3, _ = torch.max(probs3, dim=-1)
            mask = max_probs2.ge(max_probs3)
            probs3[mask, :] = probs2[mask, :]

        return probs3, domain_pred

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
                    )

        return

    def configure_model(self, model: nn.Module):
        self.classifier = get_named_submodule(model, self.classifier_name)
        set_named_submodule(model, self.classifier_name, nn.Identity())

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
                num_domains=2,
                momentum_a=self.eta,
                gamma=self.gamma,
                dynamic=True,
                self_training=False,
                pruning=self.prune,
                max_domains=self.max_domains,
                dist_metric=self.dist_metric,
            )

            if name == self.layer:
                print(f"configure_domain_prediction: {name}")
                momentum_bn.do_predict_domain = True
                momentum_bn.mode = self.mode_domain_prediction

            set_named_submodule(model, name, momentum_bn)

        return model
