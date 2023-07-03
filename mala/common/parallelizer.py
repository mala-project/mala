"""Functions for operating MALA in parallel."""
from collections import defaultdict
import platform
import warnings

try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    pass
import torch

use_horovod = False
use_mpi = False
comm = None
local_mpi_rank = None
current_verbosity = 0
lammps_instance = None


def set_current_verbosity(new_value):
    """
    Set the verbosity used for the printout statements.

    Should only be called by the parameters file, not by the user directly!

    Parameters
    ----------
    new_value : int
        New verbosity.
    """
    global current_verbosity
    current_verbosity = new_value


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
        from mpi4py import MPI

        global comm
        comm = MPI.COMM_WORLD

    # else:
    #     global comm
    #     comm = MockCommunicator()


def set_lammps_instance(new_instance):
    """
    Set a new LAMMPS instance to be targeted during the finalize call.

    This currently has to be done in order for Kokkos to not through an
    error when operating in GPU descriptor calculation mode.

    Parameters
    ----------
    new_instance : lammps.LAMMPS
        A LAMMPS instance currently in memory to be properly finalized at the
        end of the script.

    """
    import lammps
    global lammps_instance
    if isinstance(new_instance, lammps.core.lammps):
        lammps_instance = new_instance


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
    used on one node. Always returns 0 in the serial case.

    Originally obtained from:
    https://github.com/hiwonjoon/ICML2019-TREX/blob/master/mujoco/learner/baselines/baselines/common/mpi_util.py

    License:
    MIT License

    Copyright (c) 2019 Daniel Brown and Wonjoon Goo

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
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


def barrier():
    """General interface for a barrier."""
    if use_horovod:
        hvd.allreduce(torch.tensor(0), name='barrier')
    if use_mpi:
        comm.Barrier()
    return


def printout(*values, sep=' ', min_verbosity=0):
    """
    Interface to built-in "print" for parallel runs. Can be used like print.

    Linked to the verbosity option in parameters. By default, all messages are
    treated as high level messages and will be printed.

    Parameters
    ----------
    values
        Values to be printed.

    sep : string
        Separator between printed values.

    min_verbosity : int
        Minimum number of verbosity for this output to still be printed.
    """
    if current_verbosity >= min_verbosity:
        outstring = sep.join([str(v) for v in values])
        if get_rank() == 0:
            print(outstring)


def parallel_warn(warning, min_verbosity=0, category=UserWarning):
    """
    Interface for warnings in parallel runs. Can be used like warnings.warn.

    Linked to the verbosity option in parameters. By default, all messages are
    treated as high level messages and will be printed.

    Parameters
    ----------
    warning
        Warning to be printed.
    min_verbosity : int
        Minimum number of verbosity for this output to still be printed.

    category : class
        Category of the warning to be thrown.
    """
    if current_verbosity >= min_verbosity:
        if get_rank() == 0:
            warnings.warn(warning, category=category)


def finalize():
    """Properly shut down lingering Kokkos/GPU instances."""
    if lammps_instance is not None:
        lammps_instance.finalize()
