import torch


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


