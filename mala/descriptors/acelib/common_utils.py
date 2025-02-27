import numpy as np
from sympy.combinatorics import Permutation


def filled_perm(tups, rank):
    allinds = list(range(rank))
    try:
        remainders = [i for i in allinds if i not in flatten(tups)]
        alltups = tups + tuple([tuple([k]) for k in remainders])
    except TypeError:
        remainders = [i for i in allinds if i not in flatten(flatten(tups))]
        alltups = tups + tuple([tuple([k]) for k in remainders])
    return Permutation(alltups)


def flatten(lstoflsts):
    try:
        flat = [i for sublist in lstoflsts for i in sublist]
        return flat
    except TypeError:
        return lstoflsts


def group_vec_by_node(vec, nodes, remainder=None):
    # vec_by_tups = [tuple([vec[node[0]],vec[node[1]]]) for node in nodes]
    vec_by_tups = []
    for node in nodes:
        orbit_list = []
        for inode in node:
            orbit_list.append(vec[inode])
        orbit_tup = tuple(orbit_list)
        vec_by_tups.append(orbit_tup)
    if remainder != None:
        vec_by_tups = vec_by_tups + [tuple([vec[remainder]])]
    return vec_by_tups


def group_vec_by_orbits(vec, part):
    ind_range = np.sum(part)
    assert len(vec) == ind_range, "vector must be able to fit in the partion"
    count = 0
    by_orbits = []
    for orbit in part:
        orbit_vec = []
        for i in range(orbit):
            orbit_vec.append(vec[count])
            count += 1
        by_orbits.append(tuple(orbit_vec))
    return tuple(by_orbits)


local_sigma_c_partitions = {
    1: [(1,)],
    2: [(2,)],
    3: [(2, 1)],
    4: [(4,), (2, 2)],
    5: [(4, 1), (2, 2, 1)],
    6: [(4, 2), (2, 2, 2)],
    7: [(4, 2, 1), (2, 2, 2, 1)],
    8: [(4, 4), (4, 2, 2), (2, 2, 2, 2)],
}
