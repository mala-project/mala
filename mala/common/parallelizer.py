"""Functions for safely printing in parallel."""
try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    pass
try:
    from mpi4py import MPI
except ModuleNotFoundError:
    pass


use_horovod = False
use_mpi = False
comm = None


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
    if use_horovod:
        return hvd.rank()
    if use_mpi:
        return comm.Get_rank()
    return 0

def get_size():
    if use_horovod:
        return hvd.size()
    if use_mpi:
        return comm.Get_size()

# TODO: This is hacky, improve it.
def get_comm():
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
