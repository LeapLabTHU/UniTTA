import torch
import torch.nn as nn
from .base_adapter import BaseAdapter
from ..utils.wrapper_model import WrapperModel
import logging
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AffinityMatrix:
    def __init__(self, **kwargs):
        pass

    def __call__(X, **kwargs):
        raise NotImplementedError

    def is_psd(self, mat):
        eigenvalues = torch.eig(mat)[0][:, 0].sort(descending=True)[0]
        return eigenvalues, float((mat == mat.t()).all() and (eigenvalues >= 0).all())

    def symmetrize(self, mat):
        return 1 / 2 * (mat + mat.t())


class kNN_affinity(AffinityMatrix):
    def __init__(self, knn: int, **kwargs):
        self.knn = knn

    def __call__(self, X):
        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
        n_neighbors = min(self.knn + 1, N)

        knn_index = dist.topk(n_neighbors, -1, largest=False).indices[:, 1:]  # [N, knn]

        W = torch.zeros(N, N, device=X.device)
        W.scatter_(dim=-1, index=knn_index, value=1.0)

        return W


class rbf_affinity(AffinityMatrix):
    def __init__(self, sigma: float, **kwargs):
        self.sigma = sigma
        self.k = kwargs["knn"]

    def __call__(self, X):
        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
        n_neighbors = min(self.k, N)
        kth_dist = dist.topk(k=n_neighbors, dim=-1, largest=False).values[
            :, -1
        ]  # compute k^th distance for each point, [N, knn + 1]
        sigma = kth_dist.mean()
        rbf = torch.exp(-(dist**2) / (2 * sigma**2))
        return rbf


class linear_affinity(AffinityMatrix):
    def __call__(self, X: torch.Tensor):
        """
        X: [N, d]
        """
        return torch.matmul(X, X.t())


class LAME(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        super().__init__(cfg, model, optimizer)
        self.wrapper_model = WrapperModel(model, cfg.ADAPTER.LAME.CLASSIFIER)
        self.wrapper_model.eval()

        self.knn = cfg.ADAPTER.LAME.LAME_KNN
        self.sigma = cfg.ADAPTER.LAME.LAME_SIGMA
        self.affinity = eval(f"{cfg.ADAPTER.LAME.LAME_AFFINITY}_affinity")(
            sigma=self.sigma, knn=self.knn
        )
        self.force_symmetry = cfg.ADAPTER.LAME.LAME_FORCE_SYMMETRY

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        with torch.no_grad():
            probas = self.wrapper_model(batch_data).softmax(dim=-1)  # [N, K]

            # --- Get unary and terms and kernel ---

            unary = -torch.log(probas + 1e-10)  # [N, K]

            feats = self.wrapper_model.backbone_out  # [N, d]
            feats = F.normalize(feats, p=2, dim=-1)  # [N, d]
            kernel = self.affinity(feats)  # [N, N]
            if self.force_symmetry:
                kernel = 1 / 2 * (kernel + kernel.t())
            # --- Perform optim ---
            Y = laplacian_optimization(unary, kernel)

        return Y

    def configure_model(self, model: nn.Module):
        return model


def laplacian_optimization(unary, kernel, bound_lambda=1, max_steps=100):
    E_list = []
    oldE = float("inf")
    Y = (-unary).softmax(-1)  # [N, K]
    for i in range(max_steps):
        pairwise = bound_lambda * kernel.matmul(Y)  # [N, K]
        exponent = -unary + pairwise
        Y = exponent.softmax(-1)
        E = entropy_energy(Y, unary, pairwise, bound_lambda).item()
        E_list.append(E)

        if i > 1 and (abs(E - oldE) <= 1e-8 * abs(oldE)):
            logger.info(f"Converged in {i} iterations")
            break
        else:
            oldE = E

    return Y


def entropy_energy(Y, unary, pairwise, bound_lambda):
    E = (unary * Y - bound_lambda * pairwise * Y + Y * torch.log(Y.clip(1e-20))).sum()
    return E
