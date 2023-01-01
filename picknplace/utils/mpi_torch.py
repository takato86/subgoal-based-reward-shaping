from picknplace.utils.mpi import num_procs, broadcast, mpi_avg
import torch
from picknplace.utils.constants import device


def mpi_avg_grad(module):
    """MPIプロセス間の勾配の平均化"""
    if num_procs() == 1:
        return

    for p in module.parameters():
        p_grad_numpy = p.grad.cpu().numpy()
        avg_p_grad = mpi_avg(p_grad_numpy)
        p_grad_numpy[:] = avg_p_grad[:]
        p.grad = torch.as_tensor(
            p_grad_numpy,
            dtype=torch.float32,
            device=device()
        )


def sync_params(module):
    """パラメータをMPIプロセス間で同期"""
    if num_procs() == 1:
        return

    for p in module.parameters():
        p_numpy = p.data.cpu().numpy()
        broadcast(p_numpy)
        p.data = torch.as_tensor(
            p_numpy,
            dtype=torch.float32,
            device=device()
        )
