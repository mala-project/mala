"""Useful helper functions for several files in the ACE library."""

import itertools

import numpy as np
from sympy.combinatorics import Permutation


def filled_perm(tuples, rank):
    """
    Create a filled permutation.

    ACE_DOCS_MISSING - What is this used for?

    Parameters
    ----------
    tuples : List
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
    list_of_lists : List
        List of lists.

    Returns
    -------
    list : List
        Flattened list.
    """
    try:
        flat = [i for sublist in list_of_lists for i in sublist]
        return flat
    except TypeError:
        return list_of_lists


def group_vector_by_nodes(vector, nodes, remainder=None):
    """
    Group a vector according to the mapping given in nodes.

    If there is a remainder, it is added to the end of the list.

    Parameters
    ----------
    vector : List
        Vector to be grouped.

    nodes : List
        List of nodes.

    remainder : int
        Remainder to be added to the end of the list.

    Returns
    -------
    vector_by_tuples : List
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
    vector : List
        Vector to be grouped.

    partition : List
        Partition to be grouped by.

    Returns
    -------
    vector_by_orbits : List
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


def group_vector_by_nodes_pairwise(vector, nodes, remainder=None):
    """
    Group a vector of l quantum numbers pairwise.

    Parameters
    ----------
    vector : List
        vector of l quantum numbers.

    nodes : List
        List of nodes by which to pair vector.

    remainder
        ACE_DOCS_MISSING

    Returns
    -------
    vector_by_tuples : List
    """
    vector_by_tuples = [
        tuple([vector[node[0]], vector[node[1]]]) for node in nodes
    ]

    # ACE_DOCS_MISSING - Am I missing something here that needs to be
    # documented or does remainder really do nothing?
    if remainder is not None:
        vector_by_tuples = vector_by_tuples
    return vector_by_tuples


def get_ms(l_vector, M_R=0):
    r"""
    Retrieve the set of m_i combination with \sum_i m_i = M_R.

    These combinations are retrieved for an arbitrary l vector.

    Parameters
    ----------
    l_vector : List
        Arbitrary vector.

    M_R : int
        Sum to which combinations sum.

    Returns
    -------
    m_strs : List
        Set of all combinations.
    """
    m_ranges = {
        ind: range(-l_vector[ind], l_vector[ind] + 1)
        for ind in range(len(l_vector))
    }
    m_range_arrays = [list(m_ranges[ind]) for ind in range(len(l_vector))]
    m_combos = list(itertools.product(*m_range_arrays))
    first_m_filter = [i for i in m_combos if np.sum(i) == M_R]
    m_list_replace = ["%d"] * len(l_vector)
    m_str_variable = ",".join(b for b in m_list_replace)
    m_strs = [m_str_variable % fmf for fmf in first_m_filter]
    return m_strs


def check_triangle(l1, l2, l3):
    """
    Check if a triangle can be formed with the given sides.

    Parameters
    ----------
    l1 : int
        Length of the first side.

    l2 : int
        Length of the second side.

    l3 : int
        Length of the third side.


    Returns
    -------
    condition : bool
        If True, a triangle can be formed.
    """
    lower_bound = np.abs(l1 - l2)
    upper_bound = np.abs(l1 + l2)
    condition = lower_bound <= l3 <= upper_bound
    return condition
