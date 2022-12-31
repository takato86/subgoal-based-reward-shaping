import logging
from mpi4py import MPI
import numpy as np


def proc_id():
    return MPI.COMM_WORLD.Get_rank()


def num_procs():
    return MPI.COMM_WORLD.Get_size()


def allreduce(*args, **kwargs):
    try:
        return MPI.COMM_WORLD.Allreduce(*args, **kwargs)
    except Exception as e:
        logging.error("x={}".format(args[0]))
        logging.error("buff={}".format(args[1]))
        logging.error(e)
        raise


def mpi_op(x, op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    allreduce(x, buff, op=op)
    return buff[0] if scalar else buff


def mpi_avg(x):
    return mpi_op(x, MPI.SUM) / num_procs()


def mpi_sum(x):
    return mpi_op(x, MPI.SUM)


def broadcast(x, root=0):
    MPI.COMM_WORLD.Bcast(x, root=root)
