import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_adapter import BaseAdapter
from copy import deepcopy

# from torch.nn.utils.weight_norm import WeightNorm
# from models.model import ResNetDomainNet126
from ..utils.custom_transforms import get_tta_transforms

from ..utils.losses import Entropy, SymmetricCrossEntropy, SoftLikelihoodRatio

from ..utils.unmixtns import UnMixTNS1d, UnMixTNS2d, replace_bn_layers


@torch.no_grad()
def update_model_variables(model, src_model, device, alpha=0.99):
    if alpha < 1.0:
        for param, src_param in zip(model.parameters(), src_model.parameters()):
            if param.requires_grad:
                param.data[:] = alpha * param[:].data[:] + (1 - alpha) * src_param[
                    :
                ].data[:].to(device)
    return model


@torch.no_grad()
def update_model_probs(x_ema, x, momentum=0.9):
    return momentum * x_ema + (1 - momentum) * x


class ROID(BaseAdapter):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.use_weighting = cfg.ADAPTER.ROID.USE_WEIGHTING
        self.use_prior_correction = cfg.ADAPTER.ROID.USE_PRIOR_CORRECTION
        self.use_consistency = cfg.ADAPTER.ROID.USE_CONSISTENCY
        self.momentum_src = cfg.ADAPTER.ROID.MOMENTUM_SRC
        self.momentum_probs = cfg.ADAPTER.ROID.MOMENTUM_PROBS
        self.temperature = cfg.ADAPTER.ROID.TEMPERATURE
        self.batch_size = cfg.TEST.BATCH_SIZE
        self.num_classes = cfg.CORRUPTION.NUM_CLASS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.class_probs_ema = (
            1 / self.num_classes * torch.ones(self.num_classes).to(self.device)
        )
        self.tta_transform = get_tta_transforms(cfg)

        # setup loss functions
        self.slr = SoftLikelihoodRatio()
        self.symmetric_cross_entropy = SymmetricCrossEntropy()
        self.softmax_entropy = Entropy()  # not used as loss

        # copy and freeze the source model
        # if isinstance(
        #    model, ResNetDomainNet126
        # ):  # https://github.com/pytorch/pytorch/issues/28594
        #    for module in model.modules():
        #        for _, hook in module._forward_pre_hooks.items():
        #            if isinstance(hook, WeightNorm):
        #                delattr(module, hook.name)

        # note: reduce memory consumption by only saving normalization parameters
        self.src_model = deepcopy(self.model).cpu()
        for param in self.src_model.parameters():
            param.detach_()

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.models = [self.src_model, self.model]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        imgs_test = x
        outputs = self.model(imgs_test)

        if self.use_weighting:
            with torch.no_grad():
                # calculate diversity based weight
                weights_div = 1 - F.cosine_similarity(
                    self.class_probs_ema.unsqueeze(dim=0), outputs.softmax(1), dim=1
                )
                weights_div = (weights_div - weights_div.min()) / (
                    weights_div.max() - weights_div.min()
                )
                mask = weights_div < weights_div.mean()

                # calculate certainty based weight
                weights_cert = -self.softmax_entropy(logits=outputs)
                weights_cert = (weights_cert - weights_cert.min()) / (
                    weights_cert.max() - weights_cert.min()
                )

                # calculate the final weights
                weights = torch.exp(weights_div * weights_cert / self.temperature)
                weights[mask] = 0.0

                self.class_probs_ema = update_model_probs(
                    x_ema=self.class_probs_ema,
                    x=outputs.softmax(1).mean(0),
                    momentum=self.momentum_probs,
                )

        # calculate the soft likelihood ratio loss
        loss_out = self.slr(logits=outputs)

        # weight the loss
        if self.use_weighting:
            loss_out = loss_out * weights
            loss_out = loss_out[~mask]
        loss = loss_out.sum() / self.batch_size

        # calculate the consistency loss
        if self.use_consistency:
            outputs_aug = self.model(self.tta_transform(imgs_test[~mask]))
            loss += (
                self.symmetric_cross_entropy(x=outputs_aug, x_ema=outputs[~mask])
                * weights[~mask]
            ).sum() / self.batch_size

        # update the model
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.model = update_model_variables(
            self.model, self.src_model, self.device, self.momentum_src
        )

        if self.use_prior_correction:
            prior = outputs.softmax(1).mean(0)
            smooth = max(1 / outputs.shape[0], 1 / outputs.shape[1]) / torch.max(prior)
            smoothed_prior = (prior + smooth) / (1 + smooth * outputs.shape[1])
            return outputs * smoothed_prior
        else:
            return outputs

    def reset(self):
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()
        self.class_probs_ema = (
            1 / self.num_classes * torch.ones(self.num_classes).to(self.device)
        )

    def collect_params(self, model):
        """Collect the affine scale + shift parameters from normalization layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in model.named_modules():
            if isinstance(
                m,
                (
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.LayerNorm,
                    nn.GroupNorm,
                    UnMixTNS2d,
                    UnMixTNS1d,
                ),
            ):
                for np, p in m.named_parameters():
                    if np in ["weight", "bias"] and p.requires_grad:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def configure_model(self, model):
        """Configure model."""
        if self.cfg.ADAPTER.ROID.USE_UNMIXTNS:
            print("Replacing BatchNorm layers with UnMix-TNS layers")
            replace_bn_layers(
                model,
                self.cfg.ADAPTER.ROID.UNMIXTNS.NUM_COMPONENTS,
                self.cfg.TEST.BATCH_SIZE,
                self.cfg.ADAPTER.ROID.UNMIXTNS.BATCH_SIZE_MAX,
            )

        model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        model.requires_grad_(False)  # disable grad, to (re-)enable only necessary parts
        # re-enable gradient for normalization layers
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()  # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)

            elif isinstance(m, (UnMixTNS1d, UnMixTNS2d)):
                m.train()
                m.requires_grad_(True)

        return model

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_states = [deepcopy(model.state_dict()) for model in self.models]
        optimizer_state = deepcopy(self.optimizer.state_dict())
        return model_states, optimizer_state

    def load_model_and_optimizer(self):
        """Restore the model and optimizer states from copies."""
        for model, model_state in zip(self.models, self.model_states):
            model.load_state_dict(model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
