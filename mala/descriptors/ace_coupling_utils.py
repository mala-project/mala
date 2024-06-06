"""Utility functions for calculation of ACE coupling coefficients."""

import itertools

import numpy as np

def get_ms(l, M_R=0):
    # retrieves the set of m_i combinations obeying \sum_i m_i = M_R for an arbitrary l vector
    m_ranges = {ind: range(-l[ind], l[ind] + 1) for ind in range(len(l))}
    m_range_arrays = [list(m_ranges[ind]) for ind in range(len(l))]
    m_combos = list(itertools.product(*m_range_arrays))
    first_m_filter = [i for i in m_combos if np.sum(i) == M_R]
    m_list_replace = ["%d"] * len(l)
    m_str_variable = ",".join(b for b in m_list_replace)
    m_strs = [m_str_variable % fmf for fmf in first_m_filter]
    return m_strs

def check_triangle(l1, l2, l3):
    # checks triangle condition between |l1+l2| and l3
    lower_bound = np.abs(l1 - l2)
    upper_bound = np.abs(l1 + l2)
    condition = l3 >= lower_bound and l3 <= upper_bound
    return condition

def tree(l):
    # quick construction of tree leaves
    rank = len(l)
    rngs = list(range(0, rank))
    rngs = iter(rngs)
    count = 0
    tup = []
    while count < int(rank / 2):
        c1 = next(rngs)
        c2 = next(rngs)
        tup.append((c1, c2))
        count += 1
    remainder = None
    if rank % 2 != 0:
        remainder = list(range(rank))[-1]
    return tuple(tup), remainder

# groups a vector of l quantum numbers pairwise
def vec_nodes(vec, nodes, remainder=None):
    vec_by_tups = [tuple([vec[node[0]], vec[node[1]]]) for node in nodes]
    if remainder != None:
        vec_by_tups = vec_by_tups
    return vec_by_tups

# assuming a pairwise coupling structure, build the "intermediate" angular momenta
def tree_l_inters(l, L_R=0, M_R=0):
    nodes, remainder = tree(l)
    rank = len(l)
    if rank >= 3:
        base_node_inters = {
            node: get_intermediates_w(l[node[0]], l[node[1]])
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
    else:
        raise ValueError("rank %d not implemented" % rank)

    return full_inter_tuples



def generate_l_LR(lrng, rank, L_R=0, M_R=0, use_permutations=False):
    if L_R % 2 == 0:
        # symmetric w.r.t. inversion
        inv_parity = True
    if L_R % 2 != 0:
        # odd spherical harmonics are antisymmetric w.r.t. inversion
        inv_parity = False
    lmax = max(lrng)
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
                flag = (
                    check_triangle(ltup[0], ltup[1], L_R)
                    and parity_flag
                )
                if flag:
                    ls.append(lstr % ltup)
        elif rank == 3:
            nodes, remainder = tree(list(all_ls[0]))
            for ltup in all_ls:
                inters = tree_l_inters(list(ltup), L_R=L_R)
                by_node = vec_nodes(ltup, nodes, remainder)
                for inters_i in inters:
                    li_flags = [
                        check_triangle(node[0], node[1], inter)
                        for node, inter in zip(by_node, inters_i)
                    ]
                    inter_flags = [
                        check_triangle(
                            inters_i[0], ltup[remainder], L_R
                        )
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
            nodes, remainder = tree(list(all_ls[0]))
            for ltup in all_ls:
                inters = tree_l_inters(list(ltup), L_R=L_R)
                by_node = vec_nodes(ltup, nodes, remainder)
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

    return ls

def get_intermediates(l):
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

# wrapper
def get_intermediates_w(l1, l2):
    l = [l1, l2]
    return get_intermediates(l)
