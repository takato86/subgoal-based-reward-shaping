import logging
import torch
from picknplace.utils.mpi import proc_id

logger = logging.getLogger()


p_id = proc_id()
n_devices = torch.cuda.device_count()

if torch.cuda.is_available():
    gpu_no = p_id % n_devices
    global_device = torch.device(f"cuda:{gpu_no}")
    logger.info(f"Use cuda:{gpu_no}")
else:
    global_device = torch.device("cpu")
    logger.info("Use cpu")


def set_device(local_device):
    global global_device
    global_device = local_device
    return global_device


def device():
    return global_device
