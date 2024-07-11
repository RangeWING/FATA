import torch
import torch.nn as nn
import torch.random
from einops import rearrange
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

import wandb

def rearranger(pattern):
    def _fn(x, **kwargs):
        return rearrange(x, pattern, **kwargs)
    return _fn

expand_4d = rearranger('b c -> b c 1 1')

class EMA:
    def __init__(self, p=0.9):
        self.value = None
        self.p = p
    
    def update(self, value):
        self.value = value \
            if self.value is None \
            else self.p * self.value.detach() + (1 - self.p) * value 
        return self.value
    
    def get(self):
        return self.value
    

class FAugLayer():
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.is_enabled = True

    def disable(self):
        self.is_enabled = False
    
    def enable(self):     
        self.is_enabled = True
    
    def _augment(self, x: torch.Tensor):
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        if not self.is_enabled: return x
        y = self._augment(x)
        # self._input = x.clone().detach()
        # self._output = y.clone().detach()
        return torch.cat((x, y), dim=0)
    
    def hook(self, module: nn.Module, args, output: torch.Tensor):
        if not self.is_enabled:
            return output
        
        # module._output = output.clone().detach()
        return self.forward(output)
    
    @classmethod
    def register_to(cls, layer: nn.Module|str, **kwargs):
        aug_layer = cls(**kwargs)        
        # aug_layer._target_layer = layer
        layer.register_forward_hook(aug_layer.hook)
        layer._aug_layer = aug_layer
        return layer   





class FNPPlusLayer(FAugLayer):
    def __init__(self, dim: int, sigma: float=1., sample_n: int|float=1, batch_size: int=64, plus=True) -> None:
        super().__init__(dim)
        self.sigma = sigma
        self.batch_size = batch_size
        self.sample_n = sample_n
        self.plus = plus
        self.dist_a = MultivariateNormal(torch.ones(dim, requires_grad=False), self.sigma * torch.eye(dim, requires_grad=False))
        self.var_a = EMA(p=0.95)
        self.memory = {i: [] for i in range(1000)}
        self.mem_size = 0
        self._my = None

    def push_memory(self, clss: torch.Tensor):
        for i, c in enumerate(clss):
            if len(self.memory[c.item()]) < 5:
                self.memory[c.item()].append(self._output[i].detach())
                self.mem_size += 1


    def sample_memory(self, N: int=1):
        results = []
        if self.mem_size < N:
            return None, None
        
        while len(results) < N:
            k = torch.randint(len(self.memory), (1,)).item()
            row = self.memory[k]
            if len(row) > 0:
                item = row[torch.randint(len(row), (1,)).item()]
                results.append((k, item))
        classes, items = tuple(zip(*results))
        return torch.stack(items, dim=0), torch.tensor(classes)


    def _augment(self, x: torch.Tensor):
        if not self.is_enabled: return x
    
        D = len(x.shape)
        assert D == 3 or D == 4 
        N = x.size(0)
        k = max(self.sample_n, 1)

        if D == 3:
            x = x.permute(0, 2, 1) # B, C, L
        
        if D == 4:
            mu_c = x.mean((-1, -2)) # B, C
        elif D == 3:
            mu_c = x.mean(-1) # B, C, L -> B, C
        
        if self.plus and N > 1:
            var_c = mu_c.std(dim=0) # C
            var_c = self.var_a.update(var_c)

            delta = var_c / var_c.max() # C
            delta = delta.unsqueeze(0) # 1 C
        else:
            # var_c = x.view(x.size(1), -1).std(dim=1)
            delta = 0.5

        # var_c = mu_c.var(dim=0)
        # delta = var_c / var_c.max()
        
        # var_c = self.var_a.update(var_c)

        # delta = var_c / var_c.max() # C
        # delta = delta.unsqueeze(0) # 1 C

        mu_c = mu_c.repeat(k, 1) # kB, C
    
        alpha = self.dist_a.sample((k*N,)).to(x.device).detach()  # kB, C
        beta = self.dist_a.sample((k*N,)).to(x.device).detach()  # kB, C
        
        if D == 4:
            y = expand_4d(alpha) * x.repeat(k, 1, 1, 1) + expand_4d(delta * (beta - alpha) * mu_c) #kB, C, H, W
        elif D == 3:
            y = alpha.unsqueeze(-1) * x.repeat(k, 1, 1) + (delta * (beta - alpha) * mu_c).unsqueeze(-1) #kB, C, L
            y = y.permute(0, 2, 1)
        
        if self.sample_n < 1:
            n = int(N * self.sample_n)
            i = torch.randperm(N)[:n]
            y[i] = x[i]

        return y


class FNPLayer(FNPPlusLayer):
    def __init__(self, dim: int, sigma: float=1., sample_n: int|float=1, batch_size: int=64) -> None:
        super().__init__(dim, sigma, sample_n, batch_size, plus=False)