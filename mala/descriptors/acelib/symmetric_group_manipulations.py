"""Functions for computations related to group symmetry."""

from mala.descriptors.acelib.young import (
    flatten,
)
from mala.descriptors.acelib.common_utils import (
    group_vector_by_orbits,
    local_sigma_c_partitions,
)


def leaf_filter(lperms):
    """
    Filter permutations based on 2-cycles (to reflect full binary trees).

    Parameters
    ----------
    lperms : List
        Permutations of l.

    Returns
    -------
    filtered : List
        Filtered permutations of l.
    """
    rank = len(lperms[0])
    part = local_sigma_c_partitions[rank][-1]
    filtered = []
    if rank <= 5:
        for lperm in lperms:
            grouped = group_vector_by_orbits(lperm, part)
            subgroup = [g for g in grouped if len(g) > 1]
            if tuple(sorted(subgroup)) == tuple(subgroup):
                filtered.append(lperm)
            else:
                pass
    elif rank in [6, 7]:
        for lperm in lperms:
            grouped = group_vector_by_orbits(lperm, part)
            subgroup = [g for g in grouped[:2]]
            if tuple(sorted(subgroup)) == tuple(subgroup):
                filtered.append(lperm)
            else:
                pass
    else:
        raise ValueError(
            "manual orbit construction for rank %d not implemented yet" % rank
        )

    return filtered


def find_degenerate_indices(indices_list):
    """
    Find degenarate indices in a list.

    Parameters
    ----------
    indices_list : List
        List of indices

    Returns
    -------
    degenerate_indices : dict
        Dictionary of degenerate indices.
    """
    degenerate_indices = {}
    for i in range(len(indices_list)):
        if indices_list.count(indices_list[i]) >= 1:
            if indices_list[i] not in degenerate_indices:
                degenerate_indices[indices_list[i]] = []
            degenerate_indices[indices_list[i]].append(i)
    return degenerate_indices


def check_sequential(list_to_check):
    """
    Check if a list is sequential/ordered.

    Parameters
    ----------
    list_to_check : List
        List which _may_ be sequential.

    Returns
    -------
    is_sequential : bool
        If True, list is sequential (i.e., ordered).
    """
    flags = []
    if len(list_to_check) > 1:
        for i in range(len(list_to_check) - 1):
            flags.append(list_to_check[i] + 1 == list_to_check[i + 1])
        return all(flags)
    else:
        return True


def calculate_degenerate_orbit(l):
    """
    Calculate frequency partitions of a list AS ORDERED.
    
    For example: l=[1,1,2,3] -> ((2, 1, 1), ((0, 1), (2,), (3,))).

    For example: l=[1,2,2,3] -> ((1, 2, 1), ((0,), (1, 2), (3,)))

    Parameters
    ----------
    l : List
        multiset of indices to find frequency partitions for
    Returns
    -------
    degenerate_orbit : Tuple
        returns partiton of N corresponding to frequency partition and
        the cycles of the frequency partition ((<partition>),(<cycles>)) 
    """
    degen_ind_dict = find_degenerate_indices(l)
    partition = []
    inds_per_orbit = {}
    for degenval, matching_inds in degen_ind_dict.items():
        this_orbit = tuple(matching_inds)
        partition.append(this_orbit)
        try:
            inds_per_orbit[degenval].extend(matching_inds)
        except KeyError:
            inds_per_orbit[degenval] = []
            inds_per_orbit[degenval].extend(matching_inds)
    partition = tuple(partition)
    part_tup = tuple([len(ki) for ki in partition])
    return part_tup, partition


def enforce_sorted_orbit(partition_indices):
    """
    Resorts partitions of N to be compatible in increasing cycle size.

    Also breaks apart frequency partitions that aren't even to make them compatible with ladder relationships.

    For example enforce_sorted_orbit(((1,2,3),(0,))) returns the partition (1,1,2) corresponding to the permutation compatible with ladder relationships in the pairwise coupling scheme ((0,), (1,), (2, 3))). In this case, it took the (1,2,3) cycle and broke it into (1)(2,3).

    TODO: change partition_indices to permutation_indices.

    Parameters
    ----------
    partition_indices : List
        A list (presumably of lists) of indices.

    Returns
    -------
    part_tup : Tuple
        Partition of N, sorted by increasing cycle size, that is obtained from
        the input permutation indices
    """
    rank = len(flatten(partition_indices))
    couple_ref = group_vector_by_orbits(
        list(range(rank)), local_sigma_c_partitions[rank][-1]
    )
    new_partition = []
    flag = all(
        [check_sequential(oi) and oi in couple_ref for oi in partition_indices]
    )
    if not flag:
        flags = [
            check_sequential(oi) == oi and oi in couple_ref
            for oi in partition_indices
        ]
        for iflag, orbit_flag in enumerate(flags):
            if not orbit_flag:
                symmetric_sub_orbits = []
                for couple_orb in couple_ref:
                    has_symmetric_sub_orbit = all(
                        [oo in partition_indices[iflag] for oo in couple_orb]
                    )
                    if (
                        has_symmetric_sub_orbit
                        and couple_orb not in symmetric_sub_orbits
                    ):
                        symmetric_sub_orbits.append(couple_orb)
                if len(symmetric_sub_orbits) == 0:
                    new_orbit = [
                        tuple([ki]) for ki in flatten(partition_indices[iflag])
                    ]
                else:
                    remain = [
                        tuple([ki])
                        for ki in flatten(partition_indices[iflag])
                        if ki not in flatten(symmetric_sub_orbits)
                    ]
                    new_orbit = symmetric_sub_orbits + remain
                new_partition.extend(new_orbit)
                new_partition = sorted(new_partition)
            else:
                new_partition.extend(partition_indices[iflag])
    else:
        new_partition = tuple(list(partition_indices).copy())
    part_tup = tuple([len(ki) for ki in new_partition])

    # The tuple(new_partition) never seems to be used in all the functions
    # that call this function. I'll comment it out for now, in case it is
    # important for later debugging.
    return part_tup  # , tuple(new_partition)
