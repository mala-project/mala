"""Functions for safely printing in parallel."""
try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    pass

use_horovod = False


def set_horovod_status(new_value):
    """
    Set the horovod status for printing.

    By setting the horovod status via this function it can be ensured that
    printing works in parallel. The Parameters class does that for the user.

    Parameters
    ----------
    new_value : bool
        Value the horovod status has.

    """
    global use_horovod
    use_horovod = new_value


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

    if use_horovod is False:
        print(outstring)
    else:
        if hvd.rank() == 0:
            print(outstring)
