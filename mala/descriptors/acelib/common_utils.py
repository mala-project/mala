"""Useful helper functions for several files in the ACE library."""

import numpy as np
from sympy.combinatorics import Permutation


def filled_perm(tuples, rank):
    """
    Create a filled permutation.

    ACE_DOCS_MISSING - What is this used for?

    Parameters
    ----------
    tuples : list
        List of tuples.

    rank : int
        Rank of the permutation.

    Returns
    -------
    Permutation : sympy.combinatorics.Permutation
        Permutation object.
    """
    allinds = list(range(rank))
    try:
        remainders = [i for i in allinds if i not in flatten(tuples)]
        alltups = tuples + tuple([tuple([k]) for k in remainders])
    except TypeError:
        remainders = [i for i in allinds if i not in flatten(flatten(tuples))]
        alltups = tuples + tuple([tuple([k]) for k in remainders])
    return Permutation(alltups)


def flatten(list_of_lists):
    """
    Flatten a list of lists.

    Returns input, if input is not a list of lists.

    Parameters
    ----------
    list_of_lists : list
        List of lists.

    Returns
    -------
    list : list
        Flattened list.
    """
    try:
        flat = [i for sublist in list_of_lists for i in sublist]
        return flat
    except TypeError:
        return list_of_lists


def group_vector_by_node(vector, nodes, remainder=None):
    """
    Group a vector according to the mapping given in nodes.

    If there is a remainder, it is added to the end of the list.

    Parameters
    ----------
    vector : list
        Vector to be grouped.

    nodes : list
        List of nodes.

    remainder : int
        Remainder to be added to the end of the list.

    Returns
    -------
    vector_by_tuples : list
        Grouped vector.
    """
    # vector_by_tuples = [tuple([vec[node[0]],vec[node[1]]]) for node in nodes]
    vector_by_tuples = []
    for node in nodes:
        orbit_list = []
        for inode in node:
            orbit_list.append(vector[inode])
        orbit_tup = tuple(orbit_list)
        vector_by_tuples.append(orbit_tup)
    if remainder is not None:
        vector_by_tuples = vector_by_tuples + [tuple([vector[remainder]])]
    return vector_by_tuples


def group_vector_by_orbits(vector, partition):
    """
    Group vector by orbits.

    The partition is a list of integers that sum to the length of the vector.

    Parameters
    ----------
    vector : list
        Vector to be grouped.

    partition : list
        Partition to be grouped by.

    Returns
    -------
    vector_by_orbits : list
        Grouped vector.
    """
    ind_range = np.sum(partition)
    assert (
        len(vector) == ind_range
    ), "vector must be able to fit in the partion"
    count = 0
    by_orbits = []
    for orbit in partition:
        orbit_vec = []
        for i in range(orbit):
            orbit_vec.append(vector[count])
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
