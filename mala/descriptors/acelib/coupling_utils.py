"""
Utility functions for calculation of ACE coupling coefficients.

Some of the functions here are specifically related to the PA basis set,
and formerly lived in the pa_gen.py or pa_lib.py files, before these
were unified.
"""

from collections import Counter
import itertools
import json
import os

import numpy as np
from sympy.combinatorics import Permutation

from mala.descriptors.acelib.young import (
    YoungSubgroup,
)
from mala.descriptors.acelib.common_utils import (
    group_vector_by_nodes,
    local_sigma_c_partitions,
    filled_perm,
    group_vector_by_nodes_pairwise,
    check_triangle,
)
from mala.descriptors.acelib.symmetric_group_manipulations import (
    leaf_filter,
    calculate_degenerate_orbit,
    enforce_sorted_orbit,
)
from mala.descriptors.acelib.tree_sorting import (
    build_full_tree,
    build_quick_tree,
)


def build_tree_for_l_intermediates(l, L_R=0):
    """
    Build the "intermediate" angular momenta. When coupling N quantum angular momenta (and reducing product of N spherical harmonics in ACE), only 2 quantum angular momenta can be added at a time with traditional clebsch-gordan coefficients.

    Any more, and they must be reduced to an 'intermediate', and the intermediates added. The allowed intermediates are those determined by repeated quantum angular momentum addition rules (a.k.a. triangle conditions).

    This function gets all possible sets of intermediates given a set of N angular momentum quantum numbers. This is described in more detail in https://doi.org/10.1016/j.jcp.2024.113073.
    
    More detail about the intermediates may be found following Eq. 7 in the above reference.

    Parameters
    ----------
    l : List
        list (multiset) of angular momentum indices l1,l2,...lN. These correspond
        to the N spherical harmonics in the ACE descriptor(s).

    L_R : int
        Resultant angular momentum quantum number. This determines the equivariant
        character of the rank N descriptor after reduction. L_R=0 corresponds to
        a rotationally invariant feature, L_R=1 corresponds to a feature that
        transforms like a vector, L_R=2 a tensor, etc.

    Returns
    -------
    full_inter_tuples : List
        List of all possible "intermediate" angular momenta allowed by polygon
        conditions. See discussion following Eq. 7 in above reference.
    """
    nodes, remainder = build_quick_tree(l)
    rank = len(l)
    if rank >= 3:
        base_node_inters = {
            node: compute_intermediates(l[node[0]], l[node[1]])
            for node in nodes
        }

    full_inter_tuples = []

    if rank == 1:
        full_inter_tuples.append(())
    elif rank == 2:
        full_inter_tuples.append(())
    elif rank == 3:
        L1s = [i for i in base_node_inters[nodes[0]]]
        for L1 in L1s:
            if check_triangle(l[remainder], L1, L_R):
                full_inter_tuples.append(tuple([L1]))
    elif rank == 4:
        L1L2_prod = [
            i
            for i in itertools.product(
                base_node_inters[nodes[0]], base_node_inters[nodes[1]]
            )
        ]
        for L1L2 in L1L2_prod:
            L1, L2 = L1L2
            if check_triangle(L1, L2, L_R):
                good_tuple = (L1, L2)
                full_inter_tuples.append(good_tuple)
    elif rank == 5:
        L1L2_prod = [
            i
            for i in itertools.product(
                base_node_inters[nodes[0]], base_node_inters[nodes[1]]
            )
        ]
        next_node_inters = [
            compute_intermediates(L1L2[0], L1L2[1]) for L1L2 in L1L2_prod
        ]
        for L1L2, L3l in zip(L1L2_prod, next_node_inters):
            L1L2L3s = list(itertools.product([L1L2], L3l))
            for L1L2L3 in L1L2L3s:
                L1L2, L3 = L1L2L3
                L1, L2 = L1L2
                if check_triangle(l[remainder], L3, L_R):
                    good_tuple = (L1, L2, L3)
                    full_inter_tuples.append(good_tuple)
    elif rank == 6:
        L1L2L3_prod = [
            i
            for i in itertools.product(
                base_node_inters[nodes[0]],
                base_node_inters[nodes[1]],
                base_node_inters[nodes[2]],
            )
        ]
        next_node_inters = [
            compute_intermediates(L1L2L3[0], L1L2L3[1])
            for L1L2L3 in L1L2L3_prod
        ]
        for L1L2L3, L4l in zip(L1L2L3_prod, next_node_inters):
            L1L2L3L4s = list(itertools.product([L1L2L3], L4l))
            for L1L2L3L4 in L1L2L3L4s:
                L1L2L3, L4 = L1L2L3L4
                L1, L2, L3 = L1L2L3
                if check_triangle(L3, L4, L_R):
                    good_tuple = (L1, L2, L3, L4)
                    full_inter_tuples.append(good_tuple)
    elif rank == 7:
        L1L2L3_prod = [
            i
            for i in itertools.product(
                base_node_inters[nodes[0]],
                base_node_inters[nodes[1]],
                base_node_inters[nodes[2]],
            )
        ]
        next_node_inters_l = [
            compute_intermediates(L1L2L3[0], L1L2L3[1])
            for L1L2L3 in L1L2L3_prod
        ]  # left hand branch
        next_node_inters_r = [
            compute_intermediates(L1L2L3[2], l[remainder])
            for L1L2L3 in L1L2L3_prod
        ]  # right hand branch
        next_node_inters = [
            (L4, L5) for L4, L5 in zip(next_node_inters_l, next_node_inters_r)
        ]
        for L1L2L3, L45 in zip(L1L2L3_prod, next_node_inters):
            L1L2L3L4L5s = list(itertools.product([L1L2L3], *L45))
            for L1L2L3L4L5 in L1L2L3L4L5s:
                L1L2L3l, L4, L5 = L1L2L3L4L5
                L1, L2, L3 = L1L2L3l
                # L4 , L5 = L45l
                if check_triangle(L4, L5, L_R):
                    good_tuple = (L1, L2, L3, L4, L5)
                    full_inter_tuples.append(good_tuple)

    elif rank == 8:
        L1L2L3L4_prod = [
            i
            for i in itertools.product(
                base_node_inters[nodes[0]],
                base_node_inters[nodes[1]],
                base_node_inters[nodes[2]],
                base_node_inters[nodes[3]],
            )
        ]
        next_node_inters_l = [
            compute_intermediates(L1L2L3L4[0], L1L2L3L4[1])
            for L1L2L3L4 in L1L2L3L4_prod
        ]  # left hand branch
        next_node_inters_r = [
            compute_intermediates(L1L2L3L4[2], L1L2L3L4[3])
            for L1L2L3L4 in L1L2L3L4_prod
        ]  # right hand branch
        # next_node_inters = [tuple(L5+L6) for L5,L6 in zip(next_node_inters_l,
        # next_node_inters_r)]
        next_node_inters = [
            (L5, L6) for L5, L6 in zip(next_node_inters_l, next_node_inters_r)
        ]
        # print('next level',next_node_inters)
        for L1L2L3L4, L56 in zip(L1L2L3L4_prod, next_node_inters):
            L1L2L3L4L5L6s = list(itertools.product([L1L2L3L4], *L56))
            # print(L1L2L3L4L5L6s)
            # L1L2L3L4L5L6s = list(itertools.product([L1L2L3L4] , [L56]))
            for L1L2L3L4L5L6 in L1L2L3L4L5L6s:
                # L1L2L3L4l , L56l = L1L2L3L4L5L6
                L1L2L3L4l, L5, L6 = L1L2L3L4L5L6
                L1, L2, L3, L4 = L1L2L3L4l
                # L5 , L6 = L56l
                if check_triangle(L5, L6, L_R):
                    good_tuple = (L1, L2, L3, L4, L5, L6)
                    full_inter_tuples.append(good_tuple)
        # print('full inters',full_inter_tuples)
    else:
        raise ValueError("rank %d not implemented" % rank)

    return full_inter_tuples


def generate_l_LR(lrng, rank, L_R=0, M_R=0, use_permutations=False):
    """
    Generate the possible combinations of angular momentum quantum numbers for a given rank. This takes into account that the desired descriptors will, in general, be rotationally invariant.

    In short, this function enumerates all possible angular basis function indices for a given descriptor rank (before reducing according to rules defined in Eq. 33 of https://doi.org/10.1016/j.jcp.2024.113073.

    Parameters
    ----------
    lrng : List
        list of int of possible angular momentum quantum numbers. Typically, 
        these will be (0,1,2...lmax)

    rank : int
        order of the expansion, referred to as `N` in Drautz 2019, of the
        descriptors to be enumerated

    L_R : int
        Resultant angular momentum quantum number. This determines the equivariant
        character of the rank N descriptor after reduction. L_R=0 corresponds to
        a rotationally invariant feature, L_R=1 corresponds to a feature that
        transforms like a vector, L_R=2 a tensor, etc.

    M_R : int
        Resultant projection quantum number. This also determines the equivariant
        character of the rank N descriptor after reduction. M_R must obey
        -L_R <= M_R <= L_R

    use_permutations : bool
        Logical flag to generate all non-repeating permutations of `l` for

    Returns
    -------
    ls : List
        List of angular momenta.
    """
    if L_R % 2 == 0:
        # symmetric w.r.t. inversion
        inv_parity = True
    if L_R % 2 != 0:
        # odd spherical harmonics are antisymmetric w.r.t. inversion
        inv_parity = False
    ls = []

    llst = ["%d"] * rank
    lstr = ",".join(b for b in llst)

    if rank == 1:
        ls.append("%d" % L_R)

    elif rank > 1:
        all_l_perms = [b for b in itertools.product(lrng, repeat=rank)]
        if use_permutations:
            all_ls = all_l_perms.copy()
        elif not use_permutations:
            # eliminate redundant couplings by only considering lexicographically ordered l_i
            all_ls = [
                ltup for ltup in all_l_perms if ltup == tuple(sorted(ltup))
            ]
        if rank == 2:
            for ltup in all_ls:
                if inv_parity:
                    parity_flag = np.sum(ltup + (L_R,)) % 2 == 0
                elif not inv_parity:
                    parity_flag = np.sum(ltup + (L_R,)) % 2 != 0
                flag = check_triangle(ltup[0], ltup[1], L_R) and parity_flag
                if flag:
                    ls.append(lstr % ltup)
        elif rank == 3:
            nodes, remainder = build_quick_tree(list(all_ls[0]))
            for ltup in all_ls:
                inters = build_tree_for_l_intermediates(list(ltup), L_R=L_R)
                by_node = group_vector_by_nodes_pairwise(
                    ltup, nodes, remainder
                )
                for inters_i in inters:
                    li_flags = [
                        check_triangle(node[0], node[1], inter)
                        for node, inter in zip(by_node, inters_i)
                    ]
                    inter_flags = [
                        check_triangle(inters_i[0], ltup[remainder], L_R)
                    ]
                    flags = li_flags + inter_flags
                    if inv_parity:
                        parity_all = np.sum(ltup) % 2 == 0
                    elif not inv_parity:
                        parity_all = np.sum(ltup) % 2 != 0
                    if all(flags) and parity_all:
                        lsub = lstr % ltup
                        if lsub not in ls:
                            ls.append(lsub)
        elif rank == 4:
            nodes, remainder = build_quick_tree(list(all_ls[0]))
            for ltup in all_ls:
                inters = build_tree_for_l_intermediates(list(ltup), L_R=L_R)
                by_node = group_vector_by_nodes_pairwise(
                    ltup, nodes, remainder
                )
                for inters_i in inters:
                    li_flags = [
                        check_triangle(node[0], node[1], inter)
                        for node, inter in zip(by_node, inters_i)
                    ]
                    inter_flags = [
                        check_triangle(inters_i[0], inters_i[1], L_R)
                    ]
                    flags = li_flags + inter_flags
                    if inv_parity:
                        parity_all = np.sum(ltup) % 2 == 0
                    elif not inv_parity:
                        parity_all = np.sum(ltup) % 2 != 0
                    if all(flags) and parity_all:
                        lsub = lstr % ltup
                        if lsub not in ls:
                            ls.append(lsub)

        elif rank == 5:
            nodes, remainder = build_quick_tree(list(all_ls[0]))
            for ltup in all_ls:
                inters = build_tree_for_l_intermediates(list(ltup), L_R=L_R)
                by_node = group_vector_by_nodes_pairwise(
                    ltup, nodes, remainder
                )
                for inters_i in inters:
                    li_flags = [
                        check_triangle(node[0], node[1], inter)
                        for node, inter in zip(by_node, inters_i)
                    ]
                    inter_flags = [
                        check_triangle(inters_i[0], inters_i[1], inters_i[2]),
                        check_triangle(inters_i[2], ltup[remainder], L_R),
                    ]
                    flags = li_flags + inter_flags
                    if inv_parity:
                        parity_all = np.sum(ltup) % 2 == 0
                    elif not inv_parity:
                        parity_all = np.sum(ltup) % 2 != 0
                    if all(flags) and parity_all:
                        lsub = lstr % ltup
                        if lsub not in ls:
                            ls.append(lsub)

        elif rank == 6:
            nodes, remainder = build_quick_tree(list(all_ls[0]))
            for ltup in all_ls:
                inters = build_tree_for_l_intermediates(list(ltup), L_R=L_R)
                by_node = group_vector_by_nodes_pairwise(
                    ltup, nodes, remainder
                )
                for inters_i in inters:
                    li_flags = [
                        check_triangle(node[0], node[1], inter)
                        for node, inter in zip(by_node, inters_i)
                    ]
                    inter_flags = [
                        check_triangle(inters_i[0], inters_i[1], inters_i[3]),
                        check_triangle(inters_i[2], inters_i[3], L_R),
                    ]
                    flags = li_flags + inter_flags
                    if inv_parity:
                        parity_all = np.sum(ltup) % 2 == 0
                    elif not inv_parity:
                        parity_all = np.sum(ltup) % 2 != 0
                    if all(flags) and parity_all:
                        lsub = lstr % ltup
                        if lsub not in ls:
                            ls.append(lsub)

        elif rank == 7:
            nodes, remainder = build_quick_tree(list(all_ls[0]))
            for ltup in all_ls:
                inters = build_tree_for_l_intermediates(list(ltup), L_R=L_R)
                by_node = group_vector_by_nodes_pairwise(
                    ltup, nodes, remainder
                )
                for inters_i in inters:
                    li_flags = [
                        check_triangle(node[0], node[1], inter)
                        for node, inter in zip(by_node, inters_i)
                    ]
                    inter_flags = [
                        check_triangle(inters_i[0], inters_i[1], inters_i[3]),
                        check_triangle(
                            inters_i[2], ltup[remainder], inters_i[4]
                        ),
                        check_triangle(inters_i[3], inters_i[4], L_R),
                    ]
                    flags = li_flags + inter_flags
                    if inv_parity:
                        parity_all = np.sum(ltup) % 2 == 0
                    elif not inv_parity:
                        parity_all = np.sum(ltup) % 2 != 0
                    if all(flags) and parity_all:
                        lsub = lstr % ltup
                        if lsub not in ls:
                            ls.append(lsub)

        elif rank == 8:
            nodes, remainder = build_quick_tree(list(all_ls[0]))
            for ltup in all_ls:
                inters = build_tree_for_l_intermediates(list(ltup), L_R=L_R)
                by_node = group_vector_by_nodes_pairwise(
                    ltup, nodes, remainder
                )
                for inters_i in inters:
                    li_flags = [
                        check_triangle(node[0], node[1], inter)
                        for node, inter in zip(by_node, inters_i)
                    ]
                    inter_flags = [
                        check_triangle(inters_i[0], inters_i[1], inters_i[4]),
                        check_triangle(inters_i[2], inters_i[3], inters_i[5]),
                        check_triangle(inters_i[4], inters_i[5], L_R),
                    ]
                    flags = li_flags + inter_flags
                    if inv_parity:
                        parity_all = np.sum(ltup) % 2 == 0
                    elif not inv_parity:
                        parity_all = np.sum(ltup) % 2 != 0
                    if all(flags) and parity_all:
                        lsub = lstr % ltup
                        if lsub not in ls:
                            ls.append(lsub)

    return ls


def compute_intermediates(l1, l2):
    """
    Compute integers lying between absolute difference and sum of l1 and l2.
    
    This function enumerates possible third angular momentum quantum numbers that obey the triangle conditions for quantum angular momentum addition.
    
    See "Definitions" in https://doi.org/10.1016/j.jcp.2024.113073.

    Parameters
    ----------
    l1 : int
        First angular momentum quantum number

    l2 : int
        Second angular momentum quantum number

    Returns
    -------
    ints : List
        List of all integers between abs(l1-l2) and l1+l2.
    """
    l = [l1, l2]
    try:
        l = l.split(",")
        l1 = int(l[0])
        l2 = int(l[1])
    except AttributeError:
        l1 = l[0]
        l2 = l[1]

    tris = [i for i in range(abs(l1 - l2), l1 + l2 + 1)]

    ints = [i for i in tris]
    return ints


def compute_pa_labels_raw(rank, nmax, lmax, mumax, lmin=1, L_R=0, M_R=0):
    """
    Enumerate permutation-adapted ACE descriptors (ace descriptors obeying eq. 33 in https://doi.org/10.1016/j.jcp.2024.113073).

    For ranks <=3, this simply uses lexicographically ordered indices. This function enumerates all ACE descriptor labels of rank N, up to a maximumum radial index and up to a maximum angular function index (angular momentum number for spherical harmonics).
    
    Parameters
    ----------
    rank : int
        order of the expansion, referred to as `N` in Drautz 2019, of the
        descriptors to be enumerated

    nmax : any
        maximum radial basis function index for the given descriptor rank

    lmax : any
        maximum angular momentum number for the given descriptor rank (maximum angular function index)

    mumax : any
        maximum chemical basis index for the given rank (should generally be 
        mumax=len(ace_elements)

    lmin : any
        minimum angular momentum number for the given descriptor rank

    L_R : int
        Resultant angular momentum quantum number. This determines the equivariant
        character of the rank N descriptor after reduction. L_R=0 corresponds to
        a rotationally invariant feature, L_R=1 corresponds to a feature that
        transforms like a vector, L_R=2 a tensor, etc.

    M_R : int
        Resultant projection quantum number. This also determines the equivariant
        character of the rank N descriptor after reduction. M_R must obey
        -L_R <= M_R <= L_R

    Returns
    -------
    all_lammps_labels : List
        PA labels.
    """
    if rank >= 4:
        all_max_l = 12
        all_max_n = 12
        all_max_mu = 8
        label_file = os.path.join(
            os.path.dirname(__file__),
            "all_labels_mu%d_n%d_l%d_r%d.json"
            % (
                all_max_mu,
                all_max_n,
                all_max_l,
                rank,
            ),
        )
        if not os.path.isfile(label_file):
            build_and_write_to_tabulated(
                rank, all_max_n, all_max_l, label_file, L_R, M_R
            )

        with open(label_file, "r") as readjson:
            data = json.load(readjson)

        # This part does not seem to be needed at the moment.
        # lmax_strs = generate_l_LR(
        #     range(lmin, lmax + 1), rank, L_R=L_R, M_R=M_R
        # )
        # lvecs = [
        #     tuple([int(k) for k in lmax_str.split(",")])
        #     for lmax_str in lmax_strs
        # ]
        # nvecs = [i for i in itertools.combinations_with_replacement(range(0,nmax),rank)]
        muvecs = [
            i
            for i in itertools.combinations_with_replacement(
                range(mumax), rank
            )
        ]
        # reduced_nvecs=get_mapped_subset(nvecs)

        all_lammps_labs = []
        # all_not_compat = []
        possible_mus = list(range(mumax))

        lmax_strs = generate_l_LR(
            range(lmin, lmax + 1),
            rank,
            L_R=L_R,
            M_R=M_R,
            use_permutations=False,
        )
        lvecs = [
            tuple([int(k) for k in lmax_str.split(",")])
            for lmax_str in lmax_strs
        ]
        nvecs = [
            i
            for i in itertools.combinations_with_replacement(
                range(1, nmax + 1), rank
            )
        ]
        nlprd = [p for p in itertools.product(nvecs, lvecs)]

        for muvec in muvecs:
            muvec = tuple(muvec)
            # for nlblockstr in list(data['labels'].keys()):
            #    nstr,lstr = tuple(nlblockstr.split('_'))
            #    nvec = tuple([int(k) + 1 for k in nstr.split(',')])
            #    lvec = tuple([int(k) for k in lstr.split(',')])
            for nlv in nlprd:
                nvec, lvec = nlv
                nvec = tuple(nvec)
                lvec = tuple(lvec)
                # nus = from_tabulated((0,0,0,0),(1,1,1,1),(4,4,4,4),allowed_mus = possible_mus, tabulated_all = data)
                nus = read_from_tabulated(
                    muvec,
                    nvec,
                    lvec,
                    allowed_mus=possible_mus,
                    tabulated_all=data,
                )
                lammps_ready, not_compatible = lammps_remap(
                    nus, rank=rank, allowed_mus=possible_mus
                )
                all_lammps_labs.extend(lammps_ready)
                # all_not_compat.extend(not_compatible)

                # print ('raw PA-RPI',nus)
                # print ('lammps ready PA-RPI',lammps_ready)
                # print ('not compatible with lammps (PA-RPI with a nu vector that cannot be reused)',not_compatible)
    else:
        # no symmetry reduction required for rank <= 3
        # use typical lexicographical ordering for such cases
        labels = generate_nl(
            rank, nmax, lmax, mumax=mumax, lmin=lmin, L_R=L_R, all_perms=False
        )
        all_lammps_labs = labels
        # all_not_compat = []

    return all_lammps_labs


def generate_nl(rank, nmax, lmax, mumax=1, lmin=0, L_R=0, all_perms=False):
    """
    Generate lexicographically ordered n,l tuples. (useful for enumerating ACE descriptor labels up to rank 3.

    Parameters
    ----------
    rank: int
        order of the expansion, referred to as `N` in Drautz 2019, of the
        descriptors to be enumerated

    nmax : any
        maximum radial basis function index for the given descriptor rank

    lmax : any
        maximum angular momentum number for the given descriptor rank

    mumax : any
        maximum chemical basis index for the given rank (should generally be
        mumax=len(ace_elements)

    lmin : any
        minimum angular momentum number for the given descriptor rank

    L_R : int
        Resultant angular momentum quantum number. This determines the equivariant
        character of the rank N descriptor after reduction. L_R=0 corresponds to
        a rotationally invariant feature, L_R=1 corresponds to a feature that
        transforms like a vector, L_R=2 a tensor, etc.

    all_perms : bool
        logical flag to include redundant permutations of ACE descriptor labels.
        This should only be used for testing.

    Returns
    -------
    munl : List
        List of munl vectors in string format, i.e.
        mu0_mu1,mu2,...muk,n1,n2,..n_k,l1,l2,..l_k_L1-L2...-LK
    """
    munl = []

    murng = range(mumax)
    nrng = range(1, nmax + 1)
    lrng = range(lmin, lmax + 1)

    mus = create_unique_combinations(murng, rank)
    ns = create_unique_combinations(nrng, rank)
    ls = generate_l_LR(lrng, rank, L_R)

    linters_per_l = {
        l: build_tree_for_l_intermediates(
            [int(b) for b in l.split(",")], L_R=0
        )
        for l in ls
    }

    munllst = ["%d"] * int(rank * 3)
    munlstr = ",".join(b for b in munllst)
    for mu0 in murng:
        for cmbo in itertools.product(mus, ns, ls):
            mu, n, l = cmbo

            linters = linters_per_l[l]
            musplt = [int(k) for k in mu.split(",")]
            nsplt = [int(k) for k in n.split(",")]
            lsplt = [int(k) for k in l.split(",")]
            x = [(musplt[i], lsplt[i], nsplt[i]) for i in range(rank)]
            srt = sorted(x)
            if not all_perms:
                conds = x == srt
            elif all_perms:
                conds = tuple(lsplt) == tuple(sorted(lsplt))
            if conds:
                stmp = "%d_" % mu0 + munlstr % tuple(musplt + nsplt + lsplt)
                # if stmp not in munl:
                for linter in linters:
                    linter_str_lst = ["%d"] * len(linter)
                    linter_str = "-".join(b for b in linter_str_lst) % linter
                    munlL = stmp + "_" + linter_str
                    munl.append(munlL)
    munl = list(set(munl))
    return munl


def get_mapped_subset(ns):
    """
    Map n indices to a new set of indices based on the frequency of elements in n rather than the values of n themselves.

    This tool is to allow one to more conveniently order indices in their respective frequency partitions, as needed by Eq. 33 in https://doi.org/10.1016/j.jcp.2024.113073.

    Parameters
    ----------
    ns : List
        List of possible n multisets

    Returns
    -------
    reduced_ns : List
        Returns a list of n multisets that have been reordered according to the frequency
        of elements of n
    """
    mapped_n_per_n = {}
    n_per_mapped_n = {}
    for n in ns:
        n = list(n)
        unique_ns = list(set(n))
        tmpn = n.copy()
        tmpn.sort(key=Counter(n).get, reverse=True)
        unique_ns.sort()
        unique_ns.sort(key=Counter(n).get, reverse=True)
        mp_n = {unique_ns[i]: i for i in range(len(unique_ns))}
        mprev_n = {i: unique_ns[i] for i in range(len(unique_ns))}
        mappedn = [mp_n[t] for t in tmpn]
        mappedn = tuple(mappedn)
        mapped_n_per_n[tuple(n)] = mappedn
        try:
            n_per_mapped_n[mappedn].append(n)
        except KeyError:
            n_per_mapped_n[mappedn] = []
            n_per_mapped_n[mappedn].append(n)
    reduced_ns = []
    for mappedn in sorted(n_per_mapped_n.keys()):
        reduced_ns.append(tuple(n_per_mapped_n[mappedn][0]))
    return reduced_ns


def build_and_write_to_tabulated(
    rank, all_max_n, all_max_l, label_file, L_R=0, M_R=0
):
    """
    Build a tabulated PA ACE descriptor label file.

    This only matters for rank >=4. The json files build in this function are saved in the acelib directory and read in the process of computing the labels/coupling coefficients.

    Parameters
    ----------
    rank : int
        body order of angular ace descriptor labels to be generated

    L_R : int
        Resultant angular momentum quantum number. This determines the equivariant
        character of the rank N descriptor after reduction. L_R=0 corresponds to
        a rotationally invariant feature, L_R=1 corresponds to a feature that
        transforms like a vector, L_R=2 a tensor, etc.

    M_R : int
        Resultant projection quantum number. This also determines the equivariant
        character of the rank N descriptor after reduction. M_R must obey
        -L_R <= M_R <= L_R

    all_max_n : int
        max radial basis function index (with possible shift according to max chemical 
        basis index)

    all_max_l : int
        max angular basis function index

    label_file : str
        file name to contain PA labels

    Returns
    -------
    None : None
        Labels are written to file.
    """
    lmax_strs = generate_l_LR(
        range(0, all_max_l + 1), rank, L_R=L_R, M_R=M_R, use_permutations=False
    )
    lvecs = [
        tuple([int(k) for k in lmax_str.split(",")]) for lmax_str in lmax_strs
    ]
    nvecs = [
        i
        for i in itertools.combinations_with_replacement(
            range(0, all_max_n), rank
        )
    ]
    reduced_nvecs = get_mapped_subset(nvecs)

    all_PA_tabulated = []
    PA_per_nlblock = {}
    for nin in reduced_nvecs:
        for lin in lvecs:
            max_labs, all_labs, labels_per_block, original_spans = (
                generate_tree_labels(nin, lin)
            )
            combined_labs = combine_blocks(labels_per_block, lin)
            nl = (nin, lin)
            lspan_perm = list(original_spans.keys())[0]
            parity_span = [
                p
                for p in original_spans[lspan_perm]
                if np.sum(lspan_perm[:2] + p[2][:1]) % 2 == 0
                and np.sum(lspan_perm[2:4] + p[2][1:2]) % 2 == 0
            ]
            PA_labels = apply_ladder_relationships(
                lin,
                nin,
                combined_labs,
                parity_span,
                parity_span_labs=max_labs,
                full_span=original_spans[lspan_perm],
            )
            nstrlst = ["%d"] * rank
            lstrlst = ["%d"] * rank
            Lstrlst = ["%d"] * (rank - 2)
            nl_simple_labs = []
            nlstr = (
                ",".join(nstrlst) % tuple(nin)
                + "_"
                + ",".join(lstrlst) % tuple(lin)
            )
            for lab in PA_labels:
                mu0, mu, n, l, L = calculate_mu_n_l(lab, return_L=True)
                simple_str = (
                    ",".join(nstrlst) % tuple(n)
                    + "_"
                    + ",".join(lstrlst) % tuple(l)
                    + "_"
                    + ",".join(Lstrlst) % L
                )
                all_PA_tabulated.append(simple_str)
                nl_simple_labs.append(simple_str)
            PA_per_nlblock[nlstr] = nl_simple_labs

    dct = {"labels": PA_per_nlblock}
    with open(
        label_file,
        "w",
    ) as writejson:
        json.dump(dct, writejson, sort_keys=False, indent=2)


def read_from_tabulated(mu, n, l, allowed_mus=[0], tabulated_all=None):
    """
    Read PA ACE descriptor labels from tabulated data saved to a json file (by build_tabulated).

    Since functions are only tabulated for n-l a conversion is made to include chemical basis indices as well and make sure that they are permutation-adapted independent as well.

    Parameters
    ----------
    mu : List
        list of chemical basis indices mu1,mu2,...muN 

    n : List
        list of radial basis indices n1,n2,...nN

    l : List
        list of radial basis indices l1,l2,...lN

    allowed_mus : List
        all possible allowed chemical basis function indices. (generated by range(mumax))

    tabulated_all : dict
        optionally, pass in tabulated PA ACE descriptor labels as a dictionary

    Returns
    -------
    chem_labels : List
        Labels read from json file.
    """
    rank = len(l)
    Lveclst = ["%d"] * (rank - 2)
    vecstrlst = ["%d"] * rank
    mun_tupped = combine_muvector_nvector(mu, n)
    all_labels = []
    for mun_tup in mun_tupped:
        mappedn, mappedl, mprev_n, mprev = get_mapped(mun_tup, l)
        this_key_str = (
            ",".join(vecstrlst) % mappedn
            + "_"
            + ",".join(vecstrlst) % tuple(l)
        )
        these_labels = tabulated_all["labels"][this_key_str]
        mapped_labels = []
        # print (mappedn,this_key_str)
        for label in these_labels:
            radstr, lstr, Lstr = label.split("_")
            radvec = tuple([int(v) for v in radstr.split(",")])
            lvec = tuple([int(v) for v in lstr.split(",")])
            Lvec = tuple([int(v) for v in Lstr.split(",")])
            Lstr_std = "-".join(Lveclst) % Lvec
            remapped_radvec = [mprev_n[rdv] for rdv in radvec]
            mulab = [rdv[1] for rdv in remapped_radvec]
            nlab = [rdv[0] for rdv in remapped_radvec]
            mulab = tuple(mulab)
            nlab = tuple(nlab)
            nu = (
                ",".join(vecstrlst) % mulab
                + ","
                + ",".join(vecstrlst) % nlab
                + ","
                + ",".join(vecstrlst) % lvec
                + "_"
                + Lstr_std
            )
            # print (nu)
            mapped_labels.append(nu)
        all_labels.extend(mapped_labels)

    chem_labels = []
    for mu0 in allowed_mus:
        mu0_prefix = "%d_" % mu0
        for label in all_labels:
            chemlabel = mu0_prefix + label
            chem_labels.append(chemlabel)

    return chem_labels


def get_mapped(nin, lin):
    """
    Sort n and l multisets by frequency of occurence of elements in nin and lin.

    For nin, elements of nin are ordered according to their frequency, and a new index is assigned to elements of nin based on their frequency of occurence. The map between these two is saved.

    For lin, elements of lin are ordered according to their frequency, and a new index is assigned to elements of lin based on their frequency of occurence. The map between these two is saved.

    This function is used to avoid redundant enumeration for radial and angular function index multisets with the same frequency partitions as others.

    For example n=(1,1,2,2), l=(1,1,3,5) uses the same frequency partition as n=(2,2,3,3), l=(3,3,4,6). This function makes sure these two cases are handledwith the same frequency partition.

    For example, nin = [2,3,3,4] -> mappedn = [0,0,1,2], mprev_n = {0:3,1:2,2:4} and lin = [1,1,1,3] -> mappedl = [0,0,0,1], mprev = {0:1,1:3}.


    Parameters
    ----------
    nin : List
        radial indices to resort according to frequency and return the mapping to do so

    lin : List
        angular indices to resort according to frequency and return the mapping to do so

    Returns
    -------
    mappedn : tuple
        frequency-sorted indices for nin
    """
    N = len(lin)
    uniques = list(set(lin))
    tmp = list(lin).copy()
    tmp.sort(key=Counter(lin).get, reverse=True)
    uniques.sort()
    uniques.sort(key=Counter(tmp).get, reverse=True)
    mp = {uniques[i]: i for i in range(len(uniques))}
    mprev = {i: uniques[i] for i in range(len(uniques))}
    mappedl = [mp[t] for t in tmp]

    unique_ns = list(set(nin))
    tmpn = list(nin).copy()
    tmpn.sort(key=Counter(nin).get, reverse=True)
    unique_ns.sort()
    unique_ns.sort(key=Counter(nin).get, reverse=True)
    mp_n = {unique_ns[i]: i for i in range(len(unique_ns))}
    mprev_n = {i: unique_ns[i] for i in range(len(unique_ns))}
    mappedn = [mp_n[t] for t in tmpn]
    mappedn = tuple(mappedn)
    mappedl = tuple(mappedl)
    return mappedn, mappedl, mprev_n, mprev


def combine_muvector_nvector(mu, n):
    """
    Tuple vectors mu and n. Adds chemical basis to radial basis indices.

    Parameters
    ----------
    mu : List
        multiset of chemical basis indices

    n : List
        multiset of radial basis indices

    Returns
    -------
    tuppled : List
        combined chemical and radial basis indices
    """
    mu = sorted(mu)
    # n = sorted(n)
    umus = sorted(list(set(itertools.permutations(mu))))
    uns = sorted(list(set(itertools.permutations(n))))
    combos = [cmb for cmb in itertools.product(umus, uns)]
    tupped = [
        tuple(sorted([(ni, mui) for mui, ni in zip(*cmb)])) for cmb in combos
    ]
    tupped = list(set(tupped))
    # uniques = []
    # for tupi in tupped:
    #     nil = []
    #     muil = []
    #     for tupii in tupi:
    #         muil.append(tupii[1])
    #         nil.append(tupii[0])
    #     uniques.append(tuple([tuple(muil), tuple(nil)]))
    return tupped


def create_unique_combinations(lrng, size):
    """
    Create all unique combinations of a size from integers in a range. Useful for enumerating index multisets where repetition of indices is allowed.

    Parameters
    ----------
    lrng : range
        Range of l-values.

    size : int
        Size of combinations to be created.

    Returns
    -------
    uniques : List
        List of unique combinations.
    """
    uniques = []
    combs = itertools.combinations_with_replacement(lrng, size)
    for comb in combs:
        perms = itertools.permutations(comb)
        for p in perms:
            pstr = ",".join(str(k) for k in p)
            if pstr not in uniques:
                uniques.append(pstr)
    return uniques


def calculate_mu_n_l(nu_in, return_L=False):
    """
    Given an ACE descriptor label, nu, return the chemical basis function indices, radial basis function indices, and angular basis function indices.

    Parameters
    ----------
    nu_in : str
        ACE descriptor label in FitSNAP/LAMMPS format
        mu0_mu1,mu2...muN,n1,n2...nN,l1,l2...lN_L1-L2-...-L_{N-3}-L_{N-2}

    return_L : bool
        Flag to return multiset of intermediate angular indices

    Returns
    -------
    mu_n_l : tuple
        Tuple containing mu0, mu, n and l (and L, if return_L is True).
    """
    rank = calculate_mu_nu_rank(nu_in)
    if len(nu_in.split("_")) > 1:
        if len(nu_in.split("_")) == 2:
            nu = nu_in.split("_")[-1]
            Lstr = ""
        else:
            nu = nu_in.split("_")[1]
            Lstr = nu_in.split("_")[-1]
        mu0 = int(nu_in.split("_")[0])
        nusplt = [int(k) for k in nu.split(",")]
        mu = nusplt[:rank]
        n = nusplt[rank : 2 * rank]
        l = nusplt[2 * rank :]
        if len(Lstr) >= 1:
            L = tuple([int(k) for k in Lstr.split("-")])
        else:
            L = None
        if return_L:
            return mu0, mu, n, l, L
        else:
            return mu0, mu, n, l
    # provide option to get n,l for depricated descriptor labels
    else:
        nu = nu_in
        mu0 = 0
        mu = [0] * rank
        nusplt = [int(k) for k in nu.split(",")]
        n = nusplt[:rank]
        l = nusplt[rank : 2 * rank]
        return mu0, mu, n, l


def calculate_mu_nu_rank(nu_in):
    """
    Calculate mu-nu rank from nu. Given an ACE descriptor label in FitSNAP/LAMMPS format, return the rank of the descriptor.

    Parameters
    ----------
    nu_in : str
        ACE descriptor label in FitSNAP/LAMMPS format
        mu0_mu1,mu2...muN,n1,n2...nN,l1,l2...lN_L1-L2-...-L_{N-3}-L_{N-2}

    Returns
    -------
    mu_nu_rank : int
        Rank computed from label.
    """
    if len(nu_in.split("_")) > 1:
        assert len(nu_in.split("_")) <= 3, (
            "make sure your descriptor label is in proper format: mu0_mu1,mu2,"
            "mu3,n1,n2,n3,l1,l2,l3_L1"
        )
        nu = nu_in.split("_")[1]
        nu_splt = nu.split(",")
        return int(len(nu_splt) / 3)
    else:
        nu = nu_in
        nu_splt = nu.split(",")
        return int(len(nu_splt) / 2)


def lammps_remap(PA_labels, rank, allowed_mus=[0]):
    """
    Remap PA labels for LAMMPS. Takes (tabulated) PA labels enumerated with n and l multisets, and adds in chemical basis indices.

    In other words, this function maps munl PA labels to nl labels compatible with lammps .yace basis.

    Parameters
    ----------
    PA_labels : List
        List of PA labels to be remapped.

    rank : int
        Rank used for the remapping.

    allowed_mus : List
        Allowed mu values for the remapping.

    Returns
    -------
    remapped : tuple
        Tuple contain the remapped labels that are compatible with lammps descriptor calculators and,
        in very rare cases, labels that are not compatible.
    """
    transforms_all = {
        4: [
            ((0, 1), (2,), (3,)),
            ((0,), (1,), (2, 3)),
            ((0, 1), (2, 3)),
            ((0, 2), (1, 3)),
            ((3, 2, 0, 1),),
            ((2, 3, 1, 0),),
            ((0, 3), (1, 2)),
        ],
        5: [
            ((0, 1), (2,), (3,)),
            ((0,), (1,), (2, 3)),
            ((0, 1), (2, 3)),
            ((0, 2), (1, 3)),
            ((3, 2, 0, 1),),
            ((2, 3, 1, 0),),
            ((0, 3), (1, 2)),
        ],
    }  # correct for left vs right cycles in sympy
    leaf_to_internal_map = {
        4: {
            ((0, 1), (2,), (3,)): ((0,), (1,)),
            ((0,), (1,), (2, 3)): ((0,), (1,)),
            ((0, 1), (2, 3)): ((0,), (1,)),
            ((0, 2), (1, 3)): ((0, 1),),
            ((3, 2, 0, 1),): ((0, 1),),
            ((2, 3, 1, 0),): ((0, 1),),
            ((0, 3), (1, 2)): ((0, 1),),
        },
        5: {
            ((0, 1), (2,), (3,)): ((0,), (1,)),
            ((0,), (1,), (2, 3)): ((0,), (1,)),
            ((0, 1), (2, 3)): ((0,), (1,)),
            ((0, 2), (1, 3)): ((0, 1),),
            ((3, 2, 0, 1),): ((0, 1),),
            ((2, 3, 1, 0),): ((0, 1),),
            ((0, 3), (1, 2)): ((0, 1),),
        },
    }
    transforms = transforms_all[rank]
    as_perms = [Permutation(p) for p in transforms]

    Lveclst = ["%d"] * (rank - 2)
    vecstrlst = ["%d"] * rank

    all_nl = {mu0: [] for mu0 in allowed_mus}
    fs_labs = []
    not_compatible = []
    for lab in PA_labels:
        mu0, mu, n, l, Lraw = calculate_mu_n_l(lab, return_L=True)
        nl = (tuple(mu), tuple(n), tuple(l))
        nl_tup = tuple([(mui, ni, li) for mui, ni, li in zip(mu, n, l)])
        if nl in all_nl[mu0]:
            nlperms = [P(nl_tup) for P in as_perms]
            perm_source = {
                (
                    tuple([nli[0] for nli in nlp]),
                    tuple([nli[1] for nli in nlp]),
                    tuple([nli[2] for nli in nlp]),
                ): transform
                for nlp, transform in zip(nlperms, transforms)
            }
            # print ('perm source',perm_source)
            notins = [
                (
                    tuple([nli[0] for nli in nlp]),
                    tuple([nli[1] for nli in nlp]),
                    tuple([nli[2] for nli in nlp]),
                )
                not in all_nl[mu0]
                for nlp in nlperms
            ]
            if not any(notins):
                print("no other possible labels for LAMMPS", lab)
            added_count = 0
            nlpermsitr = iter(nlperms)
            nlp = next(nlpermsitr)
            try:
                while added_count < 1:
                    # for nlp in nlperms:
                    nlnew = (
                        tuple([nli[0] for nli in nlp]),
                        tuple([nli[1] for nli in nlp]),
                        tuple([nli[2] for nli in nlp]),
                    )
                    if nlnew not in all_nl[mu0]:
                        permtup = leaf_to_internal_map[rank][
                            perm_source[nlnew]
                        ]
                        perm_L = Permutation(filled_perm(permtup, rank - 2))(
                            Lraw
                        )
                        L = tuple(perm_L)
                        mustr = ",".join(vecstrlst) % nlnew[0]
                        nstr = ",".join(vecstrlst) % nlnew[1]
                        lstr = ",".join(vecstrlst) % nlnew[2]
                        Lstr = "-".join(Lveclst) % L
                        nustr = "%d_%s,%s,%s_%s" % (
                            mu0,
                            mustr,
                            nstr,
                            lstr,
                            Lstr,
                        )
                        all_nl[mu0].append(nlnew)
                        fs_labs.append(nustr)
                        added_count += 1
                    else:
                        nlp = next(nlpermsitr)
                        # print ('already used new nl')
                        # break
                        # print ('already used nl label for:',lab)
            except StopIteration:
                if not any(notins):
                    not_compatible.append(lab)
                else:
                    fs_labs.append(lab)
                all_nl[mu0].append(nl)
        else:
            fs_labs.append(lab)
            all_nl[mu0].append(nl)
    return fs_labs, not_compatible


def simple_parity_filter(l, inters, even=True):
    """
    Filter possible couplings according to parity of intermediates.

    Parameters
    ----------
    l : List
        collection of angular momentum quantum numbers [l1,l2,...lN]

    inters :
        possible multisets of intermediates [(L1,L2...L_{N-2}),(L1',L2',...L_{N-2})' ...]

    even : bool
        Control for which parity to filter according to. (For L_R=0 - will use 'even')

    Returns
    -------
    inters_filt : List
        Filtered multisets of intermediates obeying parity constraints.
    """
    nodes, remainder = build_quick_tree(l)
    base_ls = group_vector_by_nodes(l, nodes, remainder=remainder)
    base_ls = [list(k) for k in base_ls]
    if even:
        assert (
            np.sum(l) % 2
        ) == 0, "must have \sum{l_i} = even for even parity definition"
        if len(l) == 4:
            inters_filt = [
                i
                for i in inters
                if np.sum([i[0]] + base_ls[0]) % 2 == 0
                and np.sum([i[1]] + base_ls[1]) % 2 == 0
            ]
        else:
            if remainder is None:
                inters_filt = [
                    i
                    for i in inters
                    if all(
                        [
                            np.sum([i[ind]] + base_ls[ind]) % 2 == 0
                            for ind in range(len(base_ls))
                        ]
                    )
                ]
            else:
                inters_filt = [
                    i
                    for i in inters
                    if all(
                        [
                            np.sum([i[ind]] + base_ls[ind]) % 2 == 0
                            for ind in range(len(base_ls))
                        ]
                    )
                ]

    else:
        assert (
            np.sum(l) % 2
        ) != 0, "must have \sum{l_i} = odd for odd parity definition"
        print(
            "WARNING! You are using an odd parity tree. Check your labels to "
            "make sure this is what you want (this is for fitting vector "
            "quantities!)"
        )
        if len(l) == 4:
            inters_filt = [
                inters[ind]
                for ind, i in enumerate(base_ls)
                if np.sum([inters[ind][i]] + list(i)) % 2 != 0
            ]
        else:
            raise Exception("Odd parity not implemented for rank != 4")
    return inters_filt


def calculate_highest_coupling_representation(lp, lref):
    """
    Find the partition of N that has the biggest cycles that are multiples of 2.

    This is used to help define the recursion relationships assigned per frequency partition.

    Parameters
    ----------
    lp : List
        permutation of l indices

    lref : List
        sorted indices of l

    Returns
    -------
    highest_rep : tuple
        partition of N with maximized cycles which are powers of 2
    """
    rank = len(lp)
    coupling_reps = local_sigma_c_partitions[rank]
    ysgi = YoungSubgroup(rank)
    highest_rep = tuple([1] * rank)
    for rep in coupling_reps:
        ysgi.subgroup_fill(lref, [rep], semistandard=False)
        test_fills = ysgi.fills.copy()
        if lp not in test_fills:
            pass
        else:
            highest_rep = rep
            break
    return highest_rep


def generate_tree_labels(nin, lin, L_R=0):
    """
    Sorts nl according to frequency partitions, not necessarily lexicographically.

    This is just a special ordering of n and l multisets that is more compatible with the application of quantum angular momentum "ladder opertations".

    Parameters
    ----------
    nin : List
        input collection of radial basis function indices

    lin : List
        input collection of angular basis function indices

    L_R : int
        Resultant angular momentum quantum number. This determines the equivariant
        character of the rank N descriptor after reduction. L_R=0 corresponds to
        a rotationally invariant feature, L_R=1 corresponds to a feature that
        transforms like a vector, L_R=2 a tensor, etc.   

    Returns
    -------
    tree_labels : tuple
        Tuple containing max_labs, all_labs, labels_per_block and
        original_spans.
    """
    rank = len(lin)
    ysgi = YoungSubgroup(rank)

    if not isinstance(lin, list):
        lin = list(lin)
    if not isinstance(nin, list):
        nin = list(nin)

    # get possible unique l permutations based on degeneracy and coupling tree
    # structure
    ysgi.subgroup_fill(
        lin,
        partitions=[local_sigma_c_partitions[rank][-1]],
        max_orbit=len(local_sigma_c_partitions[rank][-1]),
        semistandard=False,
    )
    lperms = ysgi.fills.copy()
    lperms = leaf_filter(lperms)
    if rank not in [4, 8, 16, 32]:
        lperms_tmp = []
        used_hrep = []
        for lperm in lperms:
            hrep = calculate_highest_coupling_representation(
                tuple(lperm), tuple(lperms[0])
            )
            if hrep not in used_hrep:
                used_hrep.append(hrep)
                lperms_tmp.append(lperm)
            else:
                pass
        lperms = lperms_tmp
    original_joint_span = {lp: [] for lp in lperms}
    orb_nls = []

    ls = lperms.copy()
    nps_per_l = {}

    # get n permutations per l permutation
    # this could equivalently be done with a search over S_N
    for lp in ls:
        rank = len(lp)
        original_span_SO3 = build_tree_for_l_intermediates(lp)  # RI basis size
        degen_orbit, orbit_inds = calculate_degenerate_orbit(
            lp
        )  # PI basis size
        ysgi.subgroup_fill(nin, [degen_orbit], semistandard=False)
        degen_fills = ysgi.fills.copy()
        sequential_degen_orbit = enforce_sorted_orbit(orbit_inds)
        ysgi.subgroup_fill(nin, [sequential_degen_orbit], semistandard=False)
        nps_per_l[lp] = ysgi.fills.copy()
        original_joint_span[lp] = [
            (prd[0], lp, prd[1])
            for prd in itertools.product(degen_fills, original_span_SO3)
        ]

    labels_per_lperm = {}
    # build all labels (unsorted trees)
    for l in ls:
        l = list(l)
        subblock = []
        rank = len(l)
        inters = build_tree_for_l_intermediates(list(l), L_R=L_R)
        nperms = nps_per_l[tuple(l)]
        muperms = [tuple([0] * rank)]
        for inter in inters:
            if rank <= 5:
                if (
                    np.sum([inter[0]] + l[:2]) % 2 == 0
                    and np.sum([inter[1]] + l[2:4]) % 2 == 0
                ):
                    for muperm in muperms:
                        for nperm in nperms:
                            if rank == 5:
                                orb_nls.append(
                                    "0_%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d_%d-%d-%d"
                                    % (muperm + nperm + tuple(l) + inter)
                                )
                                subblock.append(
                                    "0_%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d_%d-%d-%d"
                                    % (muperm + nperm + tuple(l) + inter)
                                )
                            elif rank == 4:
                                orb_nls.append(
                                    "0_%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d_%d-%d"
                                    % (muperm + nperm + tuple(l) + inter)
                                )
                                subblock.append(
                                    "0_%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d_%d-%d"
                                    % (muperm + nperm + tuple(l) + inter)
                                )
        labels_per_lperm[tuple(l)] = subblock

    block_sizes = {key: len(val) for key, val in labels_per_lperm.items()}
    all_labs = []
    labels_per_block = {
        block: [] for block in sorted(list(block_sizes.keys()))
    }
    counts_per_block = {block: 0 for block in sorted(list(block_sizes.keys()))}

    # collect sorted trees only
    for block, labs in labels_per_lperm.items():
        used_ns = []
        used_ids = []
        for nu in labs:
            mu0, _, ntst, ltst, L = calculate_mu_n_l(nu, return_L=True)
            ltree = [(li, ni) for ni, li in zip(ntst, ltst)]  # sort first on n
            tree_i = build_full_tree(ltree, L, L_R)
            tid = tree_i.tree_id
            conds = (
                tid not in used_ids
            )  # sorting is ensured in construction of trees
            if conds:
                if tuple(ntst) not in used_ns:
                    used_ns.append(tuple(ntst))
                used_ids.append(tid)
                labels_per_block[block].append(nu)
                counts_per_block[block] += 1
                all_labs.append(nu)
            else:
                pass

    # collect labels per l permutation block
    max_labs = []
    max_count = max(list(counts_per_block.values()))
    for block, tree_labs in labels_per_block.items():
        if len(tree_labs) == max_count:
            max_labs.append(tree_labs.copy())
    max_labs = max_labs[0]

    return max_labs, all_labs, labels_per_block, original_joint_span


def combine_blocks(blocks, lin, L_R=0):
    """
    Recombine trees from multiple permutations of l.

    Combines 'blocks' of functions after rearranging l according to frequency.

    Parameters
    ----------
    blocks : dict
        labels per block (could use a new name)

    lin : List
        unique (nominally sorted) l

    L_R : int
        Resultant angular momentum quantum number. This determines the equivariant
        character of the rank N descriptor after reduction. L_R=0 corresponds to
        a rotationally invariant feature, L_R=1 corresponds to a feature that
        transforms like a vector, L_R=2 a tensor, etc.

    Returns
    -------
    combined_labs : List
        combined lL labels for the frequency partition of l (defining the 'block' of
        angular functions to work with).
    """
    rank = len(lin)
    lps = list(blocks.keys())
    blockpairs = [
        (block1, block2)
        for block1, block2 in itertools.product(lps, lps)
        if block1 != block2
    ]
    if len(blockpairs) == 0:
        blockpairs = [
            (block1, block2) for block1, block2 in itertools.product(lps, lps)
        ]
    block_map = {blockpair: None for blockpair in blockpairs}
    all_map = {blockpair: None for blockpair in blockpairs}
    L_map = {blockpair: None for blockpair in blockpairs}
    raw_perms = [p for p in itertools.permutations(list(range(rank)))]
    Ps = [Permutation(pi) for pi in raw_perms]
    for blockpair in list(block_map.keys()):
        l1i, l2i = blockpair
        is_sigma0 = l1i == lps[0]
        Pl1is = [P for P in Ps if tuple(P(list(l1i))) == l2i]
        Pl1_maxorbit_sizes = [
            max([len(k) for k in P.full_cyclic_form]) for P in Pl1is
        ]
        maxorbit_all = max(Pl1_maxorbit_sizes)
        maxorbit_ind = Pl1_maxorbit_sizes.index(maxorbit_all)
        if not is_sigma0:
            block_map[blockpair] = Pl1is[maxorbit_ind]
            all_map[blockpair] = Pl1is
        else:
            block_map[blockpair] = Permutation(
                tuple([tuple([ii]) for ii in list(range(rank))])
            )
            all_map[blockpair] = [
                Permutation(tuple([tuple([ii]) for ii in list(range(rank))]))
            ]

    for blockpair in list(block_map.keys()):
        l1i, l2i = blockpair
        inters1 = build_tree_for_l_intermediates(l1i, L_R)
        is_sigma0 = tuple(l1i) == lps[0]
        l1i = list(l1i)
        l2i = list(l2i)
        # intermediates hard coded for ramk 4 and 5 right now
        inters1 = [
            inter
            for inter in inters1
            if np.sum([inter[0]] + l1i[:2]) % 2 == 0
            and np.sum([inter[1]] + l1i[2:4]) % 2 == 0
        ]
        inters2 = build_tree_for_l_intermediates(l2i, L_R)
        inters2 = [
            inter
            for inter in inters2
            if np.sum([inter[0]] + l2i[:2]) % 2 == 0
            and np.sum([inter[1]] + l2i[2:4]) % 2 == 0
        ]
        if not is_sigma0:
            L_map[blockpair] = {L1i: L2i for L1i, L2i in zip(inters1, inters2)}
        else:
            L_map[blockpair] = {L1i: L1i for L1i, L1i in zip(inters1, inters1)}
    used_ids = []
    used_nl = []
    combined_labs = []
    super_inters_per_nl = {}
    for lp, nus in blocks.items():
        rank = len(lp)
        degen_orbit, orbit_inds = calculate_degenerate_orbit(lp)
        block_pairs = [
            blockpair
            for blockpair in list(block_map.keys())
            if blockpair[0] == tuple(lp)
        ]
        blockpair = block_pairs[0]
        # perm_map = block_map[block_pairs[0]]
        if rank == 4:
            perms_2_check = [block_map[blockpair]]
        else:
            perms_2_check = [block_map[blockpair]]
            # perms_2_check = all_map[blockpair]
        for nu in nus:
            mu0ii, muii, nii, lii, Lii = calculate_mu_n_l(nu, return_L=True)
            is_sigma0 = tuple(lii) == lps[0]
            degen_orbit, orbit_inds = calculate_degenerate_orbit(lp)
            nlii = [(niii, liii) for niii, liii in zip(nii, lii)]
            atrees = []
            for perm_map in perms_2_check:
                remapped = perm_map(nlii)
                newnii = [nliii[0] for nliii in remapped]
                newlii = [nliii[1] for nliii in remapped]
                new_Lii = L_map[blockpair][Lii]
                new_ltree = [
                    (liii, niii) for niii, liii in zip(newnii, newlii)
                ]
                tree_i = build_full_tree(new_ltree, Lii, L_R)
                # tree_i =  build_tree(new_ltree,new_Lii,L_R)
                tid = tree_i.tree_id
                atrees.append(tid)
            cond1 = not any([tid in used_ids for tid in atrees])
            if is_sigma0:
                cond2 = True
            else:
                cond2 = True

            if cond1 and cond2:
                combined_labs.append(nu)
                used_ids.append(tid)
                used_nl.append((tuple(newnii), tuple(newlii)))
                try:
                    super_inters_per_nl[(tuple(newnii), tuple(newlii))].append(
                        new_Lii
                    )
                except KeyError:
                    super_inters_per_nl[(tuple(newnii), tuple(newlii))] = [
                        new_Lii
                    ]
            else:
                pass
    return combined_labs


# apply ladder relationships
def apply_ladder_relationships(
    lin, nin, combined_labs, parity_span, parity_span_labs, full_span
):
    """
    Apply ladder relationships. From Goff 2024. For input angular function indices and radial function indices, apply ladder relationships to overcomplete set of L to remove redundant functions.

    These ladder relationships are derived from repeatedly applying raising/lowering relationships to the generalized coupling coefficients in https://doi.org/10.1016/j.jcp.2024.113073.

    Parameters
    ----------
    nin : List
        radial indices to resort according to frequency and return the mapping to do so

    lin : List
        angular indices to resort according to frequency and return the mapping to do so

    combined_labs : List
        blocks of lL generated based on frequency partition

    parity_span : List
        span of young subgroup * SO(3) after parity constraints applied

    parity_span_labs : List
        labels spanning young subgroup * SO(3) after parity constraints applied

    full_span : List
        span of full young subgroup * SO(3) group 

    Returns
    -------
    funcs : List
        reduced set of permutation-adapted functions
    """
    N = len(lin)
    uniques = list(set(lin))
    tmp = list(lin).copy()
    tmp.sort(key=Counter(lin).get, reverse=True)
    uniques.sort()
    uniques.sort(key=Counter(tmp).get, reverse=True)
    mp = {uniques[i]: i for i in range(len(uniques))}
    mappedl = [mp[t] for t in tmp]
    ysgi = YoungSubgroup(N)

    unique_ns = list(set(nin))
    tmpn = list(nin).copy()
    tmpn.sort(key=Counter(nin).get, reverse=True)
    unique_ns.sort()
    unique_ns.sort(key=Counter(nin).get, reverse=True)
    mp_n = {unique_ns[i]: i for i in range(len(unique_ns))}
    mappedn = [mp_n[t] for t in tmpn]
    mappedn = tuple(mappedn)
    mappedl = tuple(mappedl)

    max_labs = parity_span_labs.copy()
    #  based on degeneracy
    ndegen_rep, _ = calculate_degenerate_orbit(mappedn)
    ndegen_rep = list(ndegen_rep)
    ndegen_rep.sort(key=lambda x: x, reverse=True)
    ndegen_rep = tuple(ndegen_rep)
    degen_fam = (mappedl, ndegen_rep)

    all_inters = build_tree_for_l_intermediates(lin)
    even_inters = simple_parity_filter(lin, all_inters)

    if 0 in lin:
        funcs = combined_labs[: len(full_span)]

    else:
        if degen_fam == ((0, 0, 0, 0), (4,)):
            funcs = parity_span_labs[
                ::3
            ] 
        elif degen_fam == ((0, 0, 0, 0), (3, 1)):
            funcs = parity_span_labs[::3]
        elif degen_fam == ((0, 0, 0, 0), (2, 2)):
            funcs = parity_span_labs[: len(parity_span)]
        elif degen_fam == ((0, 0, 0, 0), (2, 1, 1)):
            funcs = parity_span_labs[: len(parity_span)]
        elif degen_fam == ((0, 0, 0, 0), (1, 1, 1, 1)):
            funcs = combined_labs[: len(full_span)]

        elif degen_fam == ((0, 0, 0, 1), (4,)):
            funcs = []
            recurmax = len(max_labs) / 2
            count = 0
            for lab in max_labs:
                mu0ii, muii, nii, lii, Lii = calculate_mu_n_l(
                    lab, return_L=True
                )
                lidegen_rep, l_orbit_inds = calculate_degenerate_orbit(lii)
                ysgi.subgroup_fill(
                    list(nin), [lidegen_rep], semistandard=False
                )
                degen_nfills = ysgi.fills.copy()
                if count < recurmax and tuple(nii) in degen_nfills:
                    funcs.append(lab)
                    count += 1
        elif degen_fam == ((0, 0, 0, 1), (3, 1)):
            funcs = []
            recurmax = len(max_labs) / 2
            count = 0
            for lab in max_labs:
                mu0ii, muii, nii, lii, Lii = calculate_mu_n_l(
                    lab, return_L=True
                )
                lidegen_rep, l_orbit_inds = calculate_degenerate_orbit(lii)
                ysgi.subgroup_fill(
                    list(nin), [lidegen_rep], semistandard=False
                )
                degen_nfills = ysgi.fills.copy()
                if count < recurmax and tuple(nii) in degen_nfills:
                    funcs.append(lab)
                    count += 1
        elif degen_fam == ((0, 0, 0, 1), (2, 2)):
            funcs = parity_span_labs[: len(parity_span)]
        elif degen_fam == ((0, 0, 0, 1), (2, 1, 1)):
            funcs = []
            recurmax = len(max_labs) / 2
            count = 0
            for lab in max_labs:
                mu0ii, muii, nii, lii, Lii = calculate_mu_n_l(
                    lab, return_L=True
                )
                lidegen_rep, l_orbit_inds = calculate_degenerate_orbit(lii)
                l_sequential_degen_orbit = enforce_sorted_orbit(l_orbit_inds)
                # switch to lower symmetry SN representation
                ysgi.subgroup_fill(
                    list(nin), [l_sequential_degen_orbit], semistandard=False
                )
                degen_nfills = ysgi.fills.copy()
                if count < recurmax and tuple(nii) in degen_nfills:
                    funcs.append(lab)
                    count += 1
        elif degen_fam == ((0, 0, 0, 1), (1, 1, 1, 1)):
            funcs = combined_labs[: len(full_span)]

        elif degen_fam == ((0, 0, 1, 1), (4,)):
            funcs = parity_span_labs
        elif degen_fam == ((0, 0, 1, 1), (3, 1)):
            funcs = parity_span_labs
        elif degen_fam == ((0, 0, 1, 1), (2, 2)):
            funcs = combined_labs[: len(parity_span) + len(even_inters[1:])]
        elif degen_fam == ((0, 0, 1, 1), (2, 1, 1)):
            funcs = combined_labs[
                : len(parity_span) + (2 * len(even_inters[1:]))
            ]
        elif degen_fam == ((0, 0, 1, 1), (1, 1, 1, 1)):
            funcs = combined_labs[: len(full_span)]

        elif degen_fam == ((0, 0, 1, 2), (4,)):
            funcs = parity_span_labs
        elif degen_fam == ((0, 0, 1, 2), (3, 1)):
            funcs = combined_labs[: len(parity_span) + len(even_inters[1:])]
        elif degen_fam == ((0, 0, 1, 2), (2, 2)):
            funcs = combined_labs[: len(parity_span) + len(all_inters[1:])]
        elif degen_fam == ((0, 0, 1, 2), (2, 1, 1)):
            funcs = combined_labs[
                : len(parity_span)
                + ((len(all_inters) - 1) * 2)
                + len(even_inters[1:])
            ]
        elif degen_fam == ((0, 0, 1, 2), (1, 1, 1, 1)):
            funcs = combined_labs[: len(full_span)]

        elif degen_fam[0] == (0, 1, 2, 3):
            funcs = combined_labs[: len(full_span)]

        elif degen_fam == ((0, 0, 0, 0, 0), (5,)):
            combined_labs.reverse()
            funcs = sorted(
                combined_labs[::4]
            )  # from rank 5 ladder relationship

        elif degen_fam == ((0, 0, 0, 0, 0), (4, 1)):
            combined_labs.reverse()
            funcs = sorted(combined_labs[: len(parity_span) - 4])

        elif degen_fam == ((0, 0, 0, 0, 0), (3, 2)):
            funcs = sorted(combined_labs[: len(parity_span) - 3])

        elif degen_fam == ((0, 0, 0, 0, 0), (3, 1, 1)):
            combined_labs.reverse()
            funcs = sorted(combined_labs[: len(parity_span) - 2])

        elif degen_fam == ((0, 0, 0, 0, 0), (2, 2, 1)):
            combined_labs.reverse()
            funcs = sorted(combined_labs[: len(parity_span)])

        elif degen_fam == ((0, 0, 0, 0, 0), (2, 1, 1, 1)):
            combined_labs.reverse()
            funcs = sorted(
                combined_labs[: int(len(max_labs) / len(even_inters))]
            )

        elif degen_fam == ((0, 0, 0, 0, 0), (1, 1, 1, 1, 1)):
            combined_labs.reverse()
            funcs = sorted(
                combined_labs[: int(len(max_labs) / len(even_inters))]
            )

        elif degen_fam == ((0, 0, 0, 0, 1), (5,)):
            combined_labs.reverse()
            funcs = sorted(combined_labs[: len(parity_span) - len(max_labs)])

        elif degen_fam == ((0, 0, 0, 0, 1), (4, 1)):
            combined_labs.reverse()
            funcs = sorted(combined_labs[: int(len(parity_span) / 2)])

        elif degen_fam == ((0, 0, 0, 0, 1), (3, 2)):
            combined_labs.reverse()
            funcs = sorted(combined_labs[: len(parity_span) - 1])

        elif degen_fam == ((0, 0, 0, 0, 1), (3, 1, 1)):
            funcs = combined_labs[: len(parity_span) - 1]

        elif degen_fam == ((0, 0, 0, 0, 1), (2, 2, 1)):
            funcs = combined_labs[
                : len(parity_span)
                + (2 * int(len(even_inters) / len(degen_fam[1])))
            ]

        elif degen_fam == ((0, 0, 0, 0, 1), (2, 1, 1, 1)):
            funcs = combined_labs[: len(parity_span) + (2 * len(even_inters))]

        elif degen_fam == ((0, 0, 0, 0, 1), (1, 1, 1, 1, 1)):
            funcs = combined_labs[: len(full_span)]

        elif degen_fam == ((0, 0, 0, 1, 1), (5,)):
            funcs = []
            for lab in parity_span_labs:
                mu0ii, muii, nii, lii, Lii = calculate_mu_n_l(
                    lab, return_L=True
                )
                if 0 not in Lii:
                    funcs.append(lab)

        elif degen_fam == ((0, 0, 0, 1, 1), (4, 1)):
            funcs = combined_labs[: int(len(parity_span)) - len(degen_fam[1])]

        elif degen_fam == ((0, 0, 0, 1, 1), (3, 2)):
            funcs = combined_labs[: len(parity_span) - 1]

        elif degen_fam == ((0, 0, 0, 1, 1), (3, 1, 1)):
            funcs = combined_labs[: int(len(full_span) / 2)]

        elif degen_fam == ((0, 0, 0, 1, 1), (2, 2, 1)):
            funcs = combined_labs[: int(len(max_labs) / 2) - len(even_inters)]

        elif degen_fam == ((0, 0, 0, 1, 1), (2, 1, 1, 1)):
            funcs = combined_labs[
                : int(len(max_labs) / 2) - (3 * len(even_inters))
            ]

        elif degen_fam == ((0, 0, 0, 1, 1), (1, 1, 1, 1, 1)):
            funcs = combined_labs[: len(full_span)]
        else:
            raise ValueError("No ladder relationship found!")

    return funcs
