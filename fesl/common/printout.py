try:
    import horovod.torch as hvd
except:
    pass

use_horovod = False
def set_horovod_status(new_value):
    global use_horovod
    use_horovod = new_value


def printout(*values, sep=' '):
    outstring = ''
    for v in values:
        outstring += v+sep
    if use_horovod is False:
        print(outstring)
    else:
        if hvd.rank() == 0:
            print(outstring)