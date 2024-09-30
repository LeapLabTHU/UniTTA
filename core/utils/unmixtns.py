import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy


def get_named_submodule(model, sub_name: str):
    names = sub_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)

    return module


def set_named_submodule(model, sub_name, value):
    names = sub_name.split(".")
    module = model
    for i in range(len(names)):
        if i != len(names) - 1:
            module = getattr(module, names[i])

        else:
            setattr(module, names[i], value)


class BaseNormLayer(nn.Module):
    def __init__(self, bn_layer, momentum, K) -> None:
        super().__init__()
        self.momentum = momentum
        self.K = K

        # other params
        self.eps = bn_layer.eps

        # Unmix TNS components
        self.register_buffer("running_means_mat", bn_layer.running_mean.repeat(K, 1)) # K, C
        self.register_buffer("running_vars_mat",  bn_layer.running_var.repeat(K, 1)) # K, C
        self.register_buffer("source_var", bn_layer.running_var)
        self.register_buffer("source_mean",  bn_layer.running_mean)

        # coefficient for initialization (alpha)
        coeff = 0.5

        if K == 1:
            noise = 0
            coeff = 0
        else:
            noise = math.sqrt(coeff * K / (K - 1)) * self.running_vars_mat.sqrt() \
                    * torch.randn_like(self.running_means_mat)
        
        self.running_means_mat += noise
        self.running_vars_mat = (1 - coeff) * self.running_vars_mat


        # bn weights and biases
        self.weight = deepcopy(bn_layer.weight)
        self.bias = deepcopy(bn_layer.bias)


    def clip_by_norm(self, x, max_norm=1.0):
        # x is (B, K, C)
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True) # B, K, 1
        clip_coeff = max_norm / (norm_x + 1e-6)
        clip_coeff = torch.clamp(clip_coeff, max=1.0)
        return x * clip_coeff

    def forward(self, x):
        raise NotImplementedError    


class UnMixTNS2d(BaseNormLayer):
    def __init__(self, bn_layer: nn.BatchNorm2d, momentum: float, K: int) -> None:
        super().__init__(bn_layer, momentum, K)

    def forward(self, x):
        inst_mean = torch.mean(x, dim=[2, 3], keepdim=False)  # (B, C)
        inst_var =  torch.var(x, dim=[2, 3], unbiased=False, keepdim=False)

        # compute dist w.r.t. unmix tns components
        with torch.no_grad():
            runnning_means_mat_norm = F.normalize(self.running_means_mat, dim=1) # (K, C)
            inst_mean_norm = F.normalize(inst_mean, dim=1) # (B, C)

            sim = torch.einsum("bc,kc->bk", inst_mean_norm, runnning_means_mat_norm) # (B, K)
            probs = torch.softmax(sim / 0.07, dim=1) # (B, K)

        # mean and var as weighted average with other running means  B, K, 1  1, K, C = B, K, C
        mean_mat = (1. - probs.unsqueeze(-1)) * self.running_means_mat.unsqueeze(0) + probs.unsqueeze(-1) * inst_mean.unsqueeze(1)
        
        
        # try uncomenting for better variance estimate.
        var_mat =  (1. - probs.unsqueeze(-1)) * self.running_vars_mat.unsqueeze(0)  + probs.unsqueeze(-1) * inst_var.unsqueeze(1) #+ \
                #    (1. - probs.unsqueeze(-1)) * probs.unsqueeze(-1) * (inst_mean.unsqueeze(1) - self.running_means_mat.unsqueeze(0))**2
        

        # B K C
        mean = torch.mean(mean_mat, dim=1)
        var =  torch.mean(var_mat, dim=1) + torch.mean(mean_mat**2, dim=1) - (torch.mean(mean_mat, dim=1))**2


        with torch.no_grad():
            diff_mean = inst_mean.unsqueeze(1) - self.running_means_mat.unsqueeze(0) # B K C

            self.running_means_mat = self.running_means_mat + self.momentum * \
                torch.mean(probs.unsqueeze(-1) * diff_mean,
                           dim=0)
            
            diff_var = inst_var.unsqueeze(1) - self.running_vars_mat.unsqueeze(0) # B K C

            self.running_vars_mat = self.running_vars_mat + self.momentum * \
                torch.mean(probs.unsqueeze(-1) * diff_var,
                           dim=0)
        
        # output
        b, c = mean.shape
        x = (x - mean.view(b, c, 1, 1)) / torch.sqrt(var.view(b, c, 1, 1) + self.eps)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)

        return x * weight + bias
    


class UnMixTNS1d(BaseNormLayer):
    def __init__(self, bn_layer: nn.BatchNorm1d, momentum: float, K: int) -> None:
        super().__init__(bn_layer, momentum, K)

    def forward(self, x):
        mean = self.source_mean.unsqueeze(0)
        var = self.source_var.unsqueeze(0)

        # output
        x = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1)
        bias = self.bias.view(1, -1)

        return x * weight + bias
    

def replace_bn_layers(model, K=16, bs=64, bs_max=64):
    normlayer_names = []

    for name, sub_module in model.named_modules():
        if isinstance(sub_module, nn.BatchNorm2d) or isinstance(sub_module, nn.BatchNorm1d):
            normlayer_names.append(name)
    
    for i, name in enumerate(normlayer_names):
        bn_layer = get_named_submodule(model, name)
        if isinstance(bn_layer, nn.BatchNorm2d):
            NewBN = UnMixTNS2d
        elif isinstance(bn_layer, nn.BatchNorm1d):
            NewBN = UnMixTNS1d
        else:
            raise RuntimeError()
        
        M = 1 - bn_layer.momentum
        # correct for batch_size
        M = M ** (bs / bs_max)
        M = 1 - M

        momentum_bn = NewBN(bn_layer, M, K)
        set_named_submodule(model, name, momentum_bn)
    
    return model