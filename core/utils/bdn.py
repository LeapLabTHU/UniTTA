# import torch.nn.functional as F
# from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

from .bn_layers import BalancedRobustBN2dV5


class BalancedDomainNormalization(nn.Module):
    def __init__(
        self,
        bn_layer: nn.BatchNorm2d,
        num_classes=1,
        num_domains=1,
        momentum_a=1e-01,
        gamma=0.0,
        dynamic=False,
        self_training=False,
        pruning=False,
        max_domains=64,
        dist_metric="w2",
    ):
        super().__init__()
        self.num_features = bn_layer.num_features
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.eps = bn_layer.eps
        self.momentum = momentum_a
        self.gamma = gamma
        self.dynamic = dynamic
        self.self_training = self_training
        self.pruning = pruning
        self.max_domains = max_domains
        if self.dynamic:
            self.num_domains = 2
            self.count_domain = torch.zeros(
                self.num_domains, dtype=torch.int64, device="cuda"
            )

        ori_mean = bn_layer.running_mean.detach().clone()
        ori_var = bn_layer.running_var.detach().clone()
        self.register_buffer("ori_mean", ori_mean)
        self.register_buffer("ori_var", ori_var)

        domain_mean = repeat(ori_mean, "c -> d c", d=self.num_domains).detach().clone()
        domain_var = repeat(ori_var, "c -> d c", d=self.num_domains).detach().clone()
        self.register_buffer("domain_mean", domain_mean)
        self.register_buffer("domain_var", domain_var)

        self.weight = nn.Parameter(bn_layer.weight.detach().clone())
        self.bias = nn.Parameter(bn_layer.bias.detach().clone())

        local_mean = (
            repeat(ori_mean, "c -> d c", d=self.num_classes * self.num_domains)
            .detach()
            .clone()
        )

        local_var = (
            repeat(ori_var, "c -> d c", d=self.num_classes * self.num_domains)
            .detach()
            .clone()
        )

        self.register_buffer("local_mean", local_mean)
        self.register_buffer("local_var", local_var)

        self.BBN = BalancedRobustBN2dV5(bn_layer, self.num_classes, momentum_a, gamma)

        self.label = None
        self.domain = None

        self.domain_pred = None
        self.do_predict_domain = False

        self.mode = None
        self.last_x = None
        self.do_expand_domain = False

        # wasserstein-2 distance by default
        self.dist_metric = dist_metric

        self.target_cover = None

    def forward(self, x):
        self.domain_mean = self.domain_mean.detach()
        self.domain_var = self.domain_var.detach()
        self.local_mean = self.local_mean.detach()
        self.local_var = self.local_var.detach()

    def bottom_cover_classes(self, class_counts, target_cover=0.9):
        sorted_counts, indices = torch.sort(class_counts, descending=False)
        total_samples = torch.sum(class_counts)
        covered_samples = 0
        bottom_classes_mask = torch.zeros(
            self.num_domains, dtype=torch.bool, device=class_counts.device
        )

        for count, idx in zip(sorted_counts, indices):
            covered_samples += count
            bottom_classes_mask[idx] = True
            if covered_samples / total_samples >= target_cover:
                break

        return bottom_classes_mask

    def expand_domain(self):
        self.num_domains += 1
        self.count_domain = torch.cat(
            [self.count_domain, torch.zeros(1, dtype=torch.int64, device="cuda")], dim=0
        )

        self.local_mean = torch.cat(
            [
                self.local_mean,
                repeat(self.ori_mean, "c -> d c", d=self.num_classes).detach().clone(),
            ],
            dim=0,
        )
        self.local_var = torch.cat(
            [
                self.local_var,
                repeat(self.ori_var, "c -> d c", d=self.num_classes).detach().clone(),
            ],
            dim=0,
        )

        self.domain_mean = torch.cat(
            [
                self.domain_mean,
                repeat(self.ori_mean, "c -> d c", d=1).detach().clone(),
            ],
            dim=0,
        )
        self.domain_var = torch.cat(
            [
                self.domain_var,
                repeat(self.ori_var, "c -> d c", d=1).detach().clone(),
            ],
            dim=0,
        )
        self.do_expand_domain = False

    def prune_domain(self, target_cover=0.5):
        idx_mask = torch.zeros(
            self.num_domains, dtype=torch.bool, device=self.count_domain.device
        )

        idx_mask[: int(self.num_domains * target_cover)] = True

        num_min = self.count_domain[idx_mask].min()

        pruning_mask = torch.logical_and(
            self.count_domain == num_min, idx_mask
        )  # threshold_mask & bottom_domains_mask

        mask = torch.logical_not(pruning_mask)

        if pruning_mask.sum() == 0:
            return

        self.domain_mean = self.domain_mean[mask]
        self.domain_var = self.domain_var[mask]
        self.count_domain = self.count_domain[mask]

        class_per_domain = self.num_classes
        extended_mask = mask.repeat_interleave(class_per_domain)
        self.local_mean = self.local_mean[extended_mask]
        self.local_var = self.local_var[extended_mask]

        self.num_domains -= pruning_mask.sum()

        self.do_prune_domain = False


class BalancedDomainNormalization2d(BalancedDomainNormalization):
    def forward(self, x):
        super().forward(x)

        label = self.label
        domain = self.domain

        if domain is not None:
            if label is not None:
                count_domain = torch.bincount(domain, minlength=self.num_domains)
                self.count_domain += count_domain

                if self.do_expand_domain:
                    self.expand_domain()

                update_statistics_BDN_2d(
                    self.local_mean,
                    self.local_var,
                    self.domain_mean,
                    self.domain_var,
                    self.momentum,
                    x,
                    label,
                    domain,
                    self.num_classes,
                    self.num_domains,
                )

            x = (
                x - rearrange(self.domain_mean[domain], "b c -> b c 1 1")
            ) / torch.sqrt(rearrange(self.domain_var[domain], "b c -> b c 1 1") + 1e-05)

            if self.do_prune_domain:
                self.prune_domain()

            return (
                self.BBN.weight[None, :, None, None] * x
                + self.BBN.bias[None, :, None, None]
            )

        else:
            if label is not None:
                self.BBN.label = label

                if self.do_predict_domain:
                    self.predict_domain(x.detach(), label)

                    if self.dynamic:
                        count_domain_pred = torch.bincount(
                            self.domain_pred, minlength=self.num_domains
                        )

                        count_domain = self.count_domain + count_domain_pred

                        if count_domain[-1] > 0:
                            self.do_expand_domain = True
                            if self.num_domains >= self.max_domains and self.pruning:
                                self.do_prune_domain = True

            return self.BBN(x)

    def predict_domain_single(self, x):
        b_var, b_mean = torch.var_mean(
            x, dim=[2, 3], unbiased=False, keepdim=False
        )  # (B, C)

        b_mean = rearrange(b_mean, "b c -> b 1 c")
        b_var = rearrange(b_var, "b c -> b 1 c")

        domain_means = self.domain_mean.clone()
        domain_var = self.domain_var.clone()
        domain_means = domain_means.unsqueeze(0)  # shape: (1, D, C)
        domain_var = domain_var.unsqueeze(0)  # shape: (1, D, C)

        dist_matrix = compute_dist_matrix(
            b_mean, b_var, domain_means, domain_var, self.dist_metric
        )
        return F.softmax(-dist_matrix, dim=1)

    def predict_domain_double(self, x):
        new_x = torch.zeros_like(x).to(x.device)
        new_x[1:] = x[:-1].clone()
        if self.last_x is not None:
            new_x[0] = self.last_x
        else:
            new_x[0] = x[0]

        self.last_x = x[-1].detach().clone()

        new_x = torch.cat(
            (
                rearrange(x, "b c h w -> b 1 c h w"),
                rearrange(new_x, "b c h w -> b 1 c h w"),
            ),
            dim=1,
        )

        b_var, b_mean = torch.var_mean(
            new_x, dim=[1, 3, 4], unbiased=False, keepdim=False
        )  # (B, C)
        b_mean = rearrange(b_mean, "b c -> b 1 c")
        b_var = rearrange(b_var, "b c -> b 1 c")

        domain_means = self.domain_mean.clone()
        domain_var = self.domain_var.clone()
        domain_means = domain_means.unsqueeze(0)  # shape: (1, D, C)
        domain_var = domain_var.unsqueeze(0)  # shape: (1, D, C)

        dist_matrix = compute_dist_matrix(
            b_mean, b_var, domain_means, domain_var, self.dist_metric
        )
        return F.softmax(-dist_matrix, dim=1)

    def predict_domain(self, x, label):
        if self.mode == "single":
            probs = self.predict_domain_single(x)
        elif self.mode == "double":
            probs = self.predict_domain_double(x)
        elif self.mode == "both":
            probs_single = self.predict_domain_single(x)
            max_probs_single, _ = torch.max(probs_single, dim=-1)

            probs = self.predict_domain_double(x)
            max_prob, _ = torch.max(probs, dim=-1)

            mask = max_probs_single.ge(max_prob)
            probs[mask, :] = probs_single[mask, :]

        else:
            raise NotImplementedError

        self.domain_pred = probs.argmax(dim=1)


def kl_divergence(mu1, sigma1, mu2, sigma2):
    sigma_ratio = sigma2 / sigma1
    variance_term = ((sigma1**2) + (mu1 - mu2) ** 2) / (2 * (sigma2**2))
    log_term = torch.log(sigma_ratio)
    return log_term + variance_term - 0.5


def compute_dist_matrix(b_mean, b_var, domain_means, domain_var, dist_metric):
    if dist_metric == "kl":
        dist_matrix = kl_divergence(
            b_mean, torch.sqrt(b_var), domain_means, torch.sqrt(domain_var)
        )
        dist_matrix += kl_divergence(
            domain_means, torch.sqrt(domain_var), b_mean, torch.sqrt(b_var)
        )
    elif dist_metric == "w2":
        dist_matrix = (
            torch.pow(b_mean - domain_means, 2)
            + b_var
            + domain_var
            - 2 * torch.sqrt(b_var) * torch.sqrt(domain_var)
        )
    elif dist_metric == "cos":
        dist_matrix = F.cosine_similarity(b_mean, domain_means, dim=2)
    elif dist_metric == "l2":
        dist_matrix = torch.pow(b_mean - domain_means, 2)
    elif dist_metric == "l1":
        dist_matrix = torch.abs(b_mean - domain_means)
    else:
        raise NotImplementedError

    return dist_matrix.sum(dim=2)


def update_statistics_BDN_2d(
    local_mean,
    local_var,
    domain_mean,
    domain_var,
    momentum,
    data,
    label,
    domain,
    num_classes,
    num_domains,
):
    domain_label = domain * num_classes + label

    domain_label_unique, new_domain_label, _ = torch.unique(
        domain_label, return_inverse=True, dim=0, return_counts=True
    )

    lm = local_mean[domain_label]
    lv = local_var[domain_label]

    m = momentum

    delta_pre = data - rearrange(lm, "b c -> b c 1 1")

    delta_k = torch.zeros(
        (domain_label_unique.size(0), data.size(1)), device=delta_pre.device
    )

    scatter_index = repeat(new_domain_label, "b -> b c", c=delta_pre.size(1))
    scatter_src = reduce(delta_pre, "b c h w -> b c", "mean")

    delta_k = torch.scatter_add(delta_k, 0, scatter_index, scatter_src)
    delta_k *= m

    local_mean[domain_label_unique] += delta_k

    delta_sigma_k = torch.zeros(
        (domain_label_unique.size(0), delta_pre.size(1)), device=delta_pre.device
    )
    # scatter_src = reduce(delta_pre.pow(2), 'n c h w -> n c', 'mean') - lv
    scatter_src = reduce(delta_pre.pow(2), "b c h w -> b c", "mean") - lv

    # delta_sigma_k = torch.scatter_add(delta_sigma_k, 0, scatter_index, scatter_src)
    delta_sigma_k = torch.scatter_add(delta_sigma_k, 0, scatter_index, scatter_src)
    # delta_sigma_k *= rearrange(m, 'k -> k 1')
    delta_sigma_k *= m

    local_var[domain_label_unique] -= delta_k.pow(2)
    local_var[domain_label_unique] += delta_sigma_k

    local_mean_reshape = rearrange(
        local_mean, "(d n) c -> d n c", d=num_domains, n=num_classes
    )
    local_var_reshape = rearrange(
        local_var, "(d n) c -> d n c", d=num_domains, n=num_classes
    )

    domain_mean.copy_(local_mean_reshape.mean(dim=1))
    domain_var.copy_(local_var_reshape.mean(dim=1) + local_mean_reshape.var(dim=1))

    return
