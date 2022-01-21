"""Functions for safely printing in parallel."""
try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    pass
try:
    from mpi4py import MPI
except ModuleNotFoundError:
    pass
import platform
from collections import defaultdict

use_horovod = False
use_mpi = False
comm = None
local_mpi_rank = None


def set_horovod_status(new_value):
    """
    Set the horovod status.

    By setting the horovod status via this function it can be ensured that
    printing works in parallel. The Parameters class does that for the user.

    Parameters
    ----------
    new_value : bool
        Value the horovod status has.

    """
    if use_mpi is True and new_value is True:
        raise Exception("Cannot use horovod and inference-level MPI at "
                        "the same time yet.")
    global use_horovod
    use_horovod = new_value


def set_mpi_status(new_value):
    """
    Set the MPI status.

    By setting the horovod status via this function it can be ensured that
    printing works in parallel. The Parameters class does that for the user.

    Parameters
    ----------
    new_value : bool
        Value the horovod status has.

    """
    if use_horovod is True and new_value is True:
        raise Exception("Cannot use horovod and inference-level MPI at "
                        "the same time yet.")
    global use_mpi
    use_mpi = new_value
    if use_mpi:
        global comm
        comm = MPI.COMM_WORLD

    # else:
    #     global comm
    #     comm = MockCommunicator()


def get_rank():
    """
    Get the rank of the current thread.

    Always returns 0 in the serial case.

    Returns
    -------
    rank : int
        The rank of the current thread.

    """
    if use_horovod:
        return hvd.rank()
    if use_mpi:
        return comm.Get_rank()
    return 0

def get_local_rank():
    """
    Get the local rank of the process.

    This is the rank WITHIN a node. Useful when multiple GPUs are
    used on one node.

    Originally copied from:
    https://github.com/hiwonjoon/ICML2019-TREX/blob/master/mujoco/learner/baselines/baselines/common/mpi_util.py
    """
    if use_horovod:
        return hvd.local_rank()
    if use_mpi:
        global local_mpi_rank
        if local_mpi_rank is None:
            this_node = platform.node()
            ranks_nodes = comm.allgather((comm.Get_rank(), this_node))
            node2rankssofar = defaultdict(int)
            local_rank = None
            for (rank, node) in ranks_nodes:
                if rank == comm.Get_rank():
                    local_rank = node2rankssofar[node]
                node2rankssofar[node] += 1
            assert local_rank is not None
            local_mpi_rank = local_rank
        return local_mpi_rank
    return 0

def get_size():
    """
    Get the number of ranks.

    Returns
    -------
    size : int
        The number of ranks.
    """
    if use_horovod:
        return hvd.size()
    if use_mpi:
        return comm.Get_size()

# TODO: This is hacky, improve it.
def get_comm():
    """
    Return the MPI communicator, if MPI is being used.

    Returns
    -------
    comm : MPI.COMM_WORLD
        A MPI communicator.

    """
    return comm

def printout(*values, sep=' '):
    """
    Interface to built-in "print" for parallel runs. Can be used like print.

    Parameters
    ----------
    values
        Values to be printed.

    sep : string
        Separator between printed values.
    """
    outstring = sep.join([str(v) for v in values])

    if get_rank() == 0:
        print(outstring)
