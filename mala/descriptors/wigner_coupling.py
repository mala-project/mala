"""Wigner coupling functions."""

from mala.descriptors import ace_coupling_utils as acu


def get_coupling(Wigner_3j, ldict, L_R=0, **kwargs):
    # for now we restrict up to rank 4 trees
    M_Rs = list(range(-L_R, L_R + 1))
    # generic coupling for any L_R - support must be added to call
    ranks = list(ldict.keys())
    coupling = {M_R: {rank: {} for rank in ranks} for M_R in M_Rs}

    for M_R in M_Rs:
        for rank in ranks:
            rnk = rank
            ls_per_rnk = acu.generate_l_LR(
                range(ldict[rank] + 1), rank, L_R, M_R, True
            )
            for lstr in ls_per_rnk:
                l = [int(k) for k in lstr.split(",")]
                if rank == 1:
                    decomped = rank_1_tree(Wigner_3j, l, L_R, M_R)
                    coupling[M_R][rnk][lstr] = decomped
                elif rank == 2:
                    decomped = rank_2_tree(Wigner_3j, l, L_R, M_R)
                    coupling[M_R][rnk][lstr] = decomped
                elif rank == 3:
                    decomped = rank_3_tree(Wigner_3j, l, L_R, M_R)
                    coupling[M_R][rnk][lstr] = decomped
                elif rank == 4:
                    decomped = rank_4_tree(Wigner_3j, l, L_R, M_R)
                    coupling[M_R][rnk][lstr] = decomped
                elif rank > 4:
                    raise ValueError(
                        "Cannot generate couplings for rank %d. symmetric L_R couplings up to rank 4 have been implemented"
                        % rank
                    )
    return coupling


def rank_1_tree(Wigner_3j, l, L_R=0, M_R=0):
    # no nodes for rank 1

    mstrs = acu.get_ms(l, M_R)
    full_inter_tuples = [()]
    assert l[0] == L_R, "invalid l=%d for irrep L_R = %d" % (l[0], L_R)

    decomposed = {
        full_inter_tup: {mstr: 0.0 for mstr in mstrs}
        for full_inter_tup in full_inter_tuples
    }

    for inter in full_inter_tuples:
        for mstr in mstrs:
            m_ints = [int(b) for b in mstr.split(",")]
            # m_1  = - M_R
            conds = m_ints[0] == -M_R
            if conds:
                w1 = Wigner_3j[
                    "%d,%d,%d,%d,%d,%d" % (l[0], m_ints[0], L_R, M_R, 0, 0)
                ]
                phase_power = 0
                phase = (-1) ** phase_power
                w = phase * w1

                decomposed[inter][mstr] = float(w)
    return decomposed

def rank_2_tree(Wigner_3j, l, L_R=0, M_R=0):
    nodes, remainder = acu.tree(l)
    node = nodes[0]
    mstrs = acu.get_ms(l, M_R)
    full_inter_tuples = [()]

    assert acu.check_triangle(
        l[0], l[1], L_R
    ), "invalid l=(%d,%d) for irrep L_R = %d" % (l[0], l[1], L_R)

    decomposed = {
        full_inter_tup: {mstr: 0.0 for mstr in mstrs}
        for full_inter_tup in full_inter_tuples
    }

    for inter in full_inter_tuples:
        for mstr in mstrs:
            m_ints = [int(b) for b in mstr.split(",")]
            # m_1 + m_2 = M1
            conds = (m_ints[0] + m_ints[1]) == M_R
            if conds:
                w1 = Wigner_3j[
                    "%d,%d,%d,%d,%d,%d"
                    % (l[0], m_ints[0], l[1], m_ints[1], L_R, -M_R)
                ]
                phase_power = L_R - M_R
                phase = (-1) ** phase_power
                w = phase * w1

                decomposed[inter][mstr] = w
    return decomposed

def rank_3_tree(Wigner_3j, l, L_R=0, M_R=0):
    full_inter_tuples = acu.tree_l_inters(l, L_R=L_R, M_R=M_R)
    mstrs = acu.get_ms(l, M_R)
    decomposed = {
        full_inter_tup: {mstr: 0.0 for mstr in mstrs}
        for full_inter_tup in full_inter_tuples
    }

    for inter in full_inter_tuples:
        L1 = inter[0]
        for mstr in mstrs:
            m_ints = [int(b) for b in mstr.split(",")]
            for M1 in range(-L1, L1 + 1):
                # m_1 + m_2 = M1
                # M1 + m_3 = M_R
                conds = (m_ints[0] + m_ints[1]) == M1 and (
                    M1 + m_ints[2]
                ) == M_R
                if conds:
                    w1 = Wigner_3j[
                        "%d,%d,%d,%d,%d,%d"
                        % (l[0], m_ints[0], l[1], m_ints[1], L1, -M1)
                    ]
                    w2 = Wigner_3j[
                        "%d,%d,%d,%d,%d,%d"
                        % (L1, M1, l[2], m_ints[2], L_R, -M_R)
                    ]
                    phase_power = (L1) - (M1) + (L_R - M_R)
                    phase = (-1) ** phase_power
                    w = phase * w1 * w2
                    decomposed[inter][mstr] = w
    return decomposed

def rank_4_tree(Wigner_3j, l, L_R=0, M_R=0):
    nodes, remainder = acu.tree(l)
    mstrs = acu.get_ms(l, M_R)
    full_inter_tuples = acu.tree_l_inters(l, L_R=L_R, M_R=M_R)
    decomposed = {
        full_inter_tup: {mstr: 0.0 for mstr in mstrs}
        for full_inter_tup in full_inter_tuples
    }

    for inter in full_inter_tuples:
        L1, L2 = inter
        for mstr in mstrs:
            m_ints = [int(b) for b in mstr.split(",")]
            for M1 in range(-L1, L1 + 1):
                for M2 in range(-L2, L2 + 1):
                    # m_1 + m_2 = M1
                    # m_4 + m_3 = M2
                    # M1 + M2 = M_R
                    conds = (
                        (m_ints[0] + m_ints[1]) == M1
                        and (m_ints[2] + m_ints[3]) == M2
                        and (M1 + M2) == M_R
                    )
                    if conds:
                        w1 = Wigner_3j[
                            "%d,%d,%d,%d,%d,%d"
                            % (l[0], m_ints[0], l[1], m_ints[1], L1, -M1)
                        ]
                        w2 = Wigner_3j[
                            "%d,%d,%d,%d,%d,%d"
                            % (l[2], m_ints[2], l[3], m_ints[3], L2, -M2)
                        ]
                        w3 = Wigner_3j[
                            "%d,%d,%d,%d,%d,%d"
                            % (L1, M1, L2, M2, L_R, -M_R)
                        ]
                        phase_power = (L1 + L2) - (M1 + M2) + (L_R - M_R)
                        phase = (-1) ** phase_power
                        w = phase * w1 * w2 * w3

                        decomposed[inter][mstr] = w
    return decomposed
