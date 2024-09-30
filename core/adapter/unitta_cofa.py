import torch
import torch.nn as nn
from .base_adapter import BaseAdapter
from ..utils.wrapper_model import WrapperModel


class UNITTA_COFA(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        super().__init__(cfg, model, optimizer)
        self.wrapper_model = WrapperModel(model, cfg.ADAPTER.UNITTA_COFA.CLASSIFIER)
        self.last_features = None
        self.filt = cfg.ADAPTER.UNITTA_COFA.FILT

        return

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        probas = self.wrapper_model(batch_data).softmax(dim=-1)  # [N, K]
        max_probs, _ = torch.max(probas, dim=-1)

        feats = self.wrapper_model.backbone_out  # [N, d]

        new_feats = torch.zeros_like(feats).to(feats.device)
        new_feats[1:] = (feats[1:] + feats[:-1]) / 2
        if self.last_features is not None:
            new_feats[0] = (feats[0] + self.last_features) / 2
        else:
            new_feats[0] = feats[0]

        self.last_features = feats[-1]

        new_probs = self.wrapper_model.classifier(new_feats).softmax(dim=-1)  # [N, K]
        max_new_probs, _ = torch.max(new_probs, dim=-1)

        if self.filt:
            mask = max_probs.ge(max_new_probs)
            new_probs[mask, :] = probas[mask, :]

        return new_probs

    def configure_model(self, model: nn.Module):
        model.requires_grad_(False)

        return model
