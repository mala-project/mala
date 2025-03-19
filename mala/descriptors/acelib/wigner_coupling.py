"""Wigner 3j coupling functions."""

import mala.descriptors.acelib.common_utils
from mala.descriptors.acelib import coupling_utils as ace_coupling_utils


def wigner_3j_coupling(wigner_3j_coefficients, ldict, L_R=0):
    """
    Compute generalized Wigner symbols for a given L_R. These, like the generalized Clebsch-Gordan coefficients, may be used to reduce products of arbitrary numbers of spherical harmonics.

    The approach to construct generalized Wigner symbols is manually defined for each rank for pedagogical and clarity purposes.

    TODO: Change name to generalized_wigner_coupling.

    Parameters
    ----------
    wigner_3j_coefficients : dict
        Precomputed Wigner 3j coefficients.

    ldict : dict
        Dictionary of ranks and their corresponding L values.

    L_R : int
        Resultant angular momentum quantum number. This determines the equivariant
        character of the rank N descriptor after reduction. L_R=0 corresponds to
        a rotationally invariant feature, L_R=1 corresponds to a feature that
        transforms like a vector, L_R=2 a tensor, etc.

    Returns
    -------
    coupling : dict
        Generalized Wigner symbols, {M_R:{rank:{l:{m:coupling_coefficient}}}
    """

    def _rank_1_tree(_wigner_3j_coefficients, _l, _L_R=0, _M_R=0):
        """
        Compute the coupling for rank 1.

        Parameters
        ----------
        _wigner_3j_coefficients : dict
        Precomputed Wigner 3j coefficients.

        _l : List
            list of angular momentum quantum numbers [l1]

        _L_R : int
            Resultant angular momentum quantum number. This determines the equivariant
            character of the rank N descriptor after reduction. L_R=0 corresponds to
            a rotationally invariant feature, L_R=1 corresponds to a feature that
            transforms like a vector, L_R=2 a tensor, etc.

        _M_R : int
            Resultant projection quantum number. This also determines the equivariant
            character of the rank N descriptor after reduction. M_R must obey
            -L_R <= M_R <= L_R

        Returns
        -------
        decomposed : dict
            Generalized Wigner symbol for rank 1
        """
        # no nodes for rank 1

        mstrs = mala.descriptors.acelib.common_utils.get_ms(_l, _M_R)
        full_inter_tuples = [()]
        assert _l[0] == _L_R, "invalid l=%d for irrep L_R = %d" % (_l[0], _L_R)

        decomposed = {
            full_inter_tup: {mstr: 0.0 for mstr in mstrs}
            for full_inter_tup in full_inter_tuples
        }

        for inter in full_inter_tuples:
            for mstr in mstrs:
                m_ints = [int(b) for b in mstr.split(",")]
                # m_1  = - M_R
                conds = m_ints[0] == -_M_R
                if conds:
                    w1 = _wigner_3j_coefficients[
                        "%d,%d,%d,%d,%d,%d"
                        % (_l[0], m_ints[0], _L_R, _M_R, 0, 0)
                    ]
                    phase_power = 0
                    phase = (-1) ** phase_power
                    w = phase * w1

                    decomposed[inter][mstr] = float(w)
        return decomposed

    def _rank_2_tree(_wigner_3j_coefficients, _l, _L_R=0, _M_R=0):
        """
        Compute the coupling for rank 2.

        Parameters
        ----------
        _wigner_3j_coefficients : dict
        Precomputed Wigner 3j coefficients.

        _l : List
            list of angular momentum quantum numbers [l1,l2]

        _L_R : int
            Resultant angular momentum quantum number. This determines the equivariant
            character of the rank N descriptor after reduction. L_R=0 corresponds to
            a rotationally invariant feature, L_R=1 corresponds to a feature that
            transforms like a vector, L_R=2 a tensor, etc.

        _M_R : int
            Resultant projection quantum number. This also determines the equivariant
            character of the rank N descriptor after reduction. M_R must obey
            -L_R <= M_R <= L_R

        Returns
        -------
        decomposed : dict
            Generalized Wigner symbol for rank 2
        """
        nodes, remainder = ace_coupling_utils.build_quick_tree(_l)
        node = nodes[0]
        mstrs = mala.descriptors.acelib.common_utils.get_ms(_l, _M_R)
        full_inter_tuples = [()]

        assert mala.descriptors.acelib.common_utils.check_triangle(
            _l[0], _l[1], _L_R
        ), "invalid l=(%d,%d) for irrep L_R = %d" % (_l[0], _l[1], _L_R)

        decomposed = {
            full_inter_tup: {mstr: 0.0 for mstr in mstrs}
            for full_inter_tup in full_inter_tuples
        }

        for inter in full_inter_tuples:
            for mstr in mstrs:
                m_ints = [int(b) for b in mstr.split(",")]
                # m_1 + m_2 = M1
                conds = (m_ints[0] + m_ints[1]) == _M_R
                if conds:
                    w1 = _wigner_3j_coefficients[
                        "%d,%d,%d,%d,%d,%d"
                        % (_l[0], m_ints[0], _l[1], m_ints[1], _L_R, -_M_R)
                    ]
                    phase_power = _L_R - _M_R
                    phase = (-1) ** phase_power
                    w = phase * w1

                    decomposed[inter][mstr] = w
        return decomposed

    def _rank_3_tree(_wigner_3j_coefficients, _l, _L_R=0, _M_R=0):
        """
        Compute the coupling for rank 3.

        Parameters
        ----------
        _wigner_3j_coefficients : dict
        Precomputed Wigner 3j coefficients.

        _l : List
            list of angular momentum quantum numbers [l1,l2,l3]

        _L_R : int
            Resultant angular momentum quantum number. This determines the equivariant
            character of the rank N descriptor after reduction. L_R=0 corresponds to
            a rotationally invariant feature, L_R=1 corresponds to a feature that
            transforms like a vector, L_R=2 a tensor, etc.

        _M_R : int
            Resultant projection quantum number. This also determines the equivariant
            character of the rank N descriptor after reduction. M_R must obey
            -L_R <= M_R <= L_R

        Returns
        -------
        decomposed : dict
            Generalized Wigner symbol for rank 3
        """
        full_inter_tuples = ace_coupling_utils.build_tree_for_l_intermediates(
            _l, L_R=_L_R
        )
        mstrs = mala.descriptors.acelib.common_utils.get_ms(_l, _M_R)
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
                    ) == _M_R
                    if conds:
                        w1 = _wigner_3j_coefficients[
                            "%d,%d,%d,%d,%d,%d"
                            % (_l[0], m_ints[0], _l[1], m_ints[1], L1, -M1)
                        ]
                        w2 = _wigner_3j_coefficients[
                            "%d,%d,%d,%d,%d,%d"
                            % (L1, M1, _l[2], m_ints[2], _L_R, -_M_R)
                        ]
                        phase_power = (L1) - (M1) + (_L_R - _M_R)
                        phase = (-1) ** phase_power
                        w = phase * w1 * w2
                        decomposed[inter][mstr] = w
        return decomposed

    def _rank_4_tree(_wigner_3j_coefficients, _l, _L_R=0, _M_R=0):
        """
        Compute the coupling for rank 4.

        Parameters
        ----------
        _wigner_3j_coefficients : dict
        Precomputed Wigner 3j coefficients.

        _l : List
            list of angular momentum quantum numbers [l1,l2,l3,l4]

        _L_R : int
            Resultant angular momentum quantum number. This determines the equivariant
            character of the rank N descriptor after reduction. L_R=0 corresponds to
            a rotationally invariant feature, L_R=1 corresponds to a feature that
            transforms like a vector, L_R=2 a tensor, etc.

        _M_R : int
            Resultant projection quantum number. This also determines the equivariant
            character of the rank N descriptor after reduction. M_R must obey
            -L_R <= M_R <= L_R

        Returns
        -------
        decomposed : dict
            Generalized Wigner symbol for rank 4

        """
        nodes, remainder = ace_coupling_utils.build_quick_tree(_l)
        mstrs = mala.descriptors.acelib.common_utils.get_ms(_l, _M_R)
        full_inter_tuples = ace_coupling_utils.build_tree_for_l_intermediates(
            _l, L_R=_L_R
        )
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
                            and (M1 + M2) == _M_R
                        )
                        if conds:
                            w1 = _wigner_3j_coefficients[
                                "%d,%d,%d,%d,%d,%d"
                                % (_l[0], m_ints[0], _l[1], m_ints[1], L1, -M1)
                            ]
                            w2 = _wigner_3j_coefficients[
                                "%d,%d,%d,%d,%d,%d"
                                % (_l[2], m_ints[2], _l[3], m_ints[3], L2, -M2)
                            ]
                            w3 = _wigner_3j_coefficients[
                                "%d,%d,%d,%d,%d,%d"
                                % (L1, M1, L2, M2, _L_R, -_M_R)
                            ]
                            phase_power = (L1 + L2) - (M1 + M2) + (_L_R - _M_R)
                            phase = (-1) ** phase_power
                            w = phase * w1 * w2 * w3

                            decomposed[inter][mstr] = w
        return decomposed

    # for now we restrict up to rank 4 trees
    M_Rs = list(range(-L_R, L_R + 1))
    # generic coupling for any L_R - support must be added to call
    ranks = list(ldict.keys())
    coupling = {M_R: {rank: {} for rank in ranks} for M_R in M_Rs}

    for M_R in M_Rs:
        for rank in ranks:
            rnk = rank
            ls_per_rnk = ace_coupling_utils.generate_l_LR(
                range(ldict[rank] + 1), rank, L_R, M_R, True
            )
            for lstr in ls_per_rnk:
                l = [int(k) for k in lstr.split(",")]
                if rank == 1:
                    decomped = _rank_1_tree(
                        wigner_3j_coefficients, l, L_R, M_R
                    )
                    coupling[M_R][rnk][lstr] = decomped
                elif rank == 2:
                    decomped = _rank_2_tree(
                        wigner_3j_coefficients, l, L_R, M_R
                    )
                    coupling[M_R][rnk][lstr] = decomped
                elif rank == 3:
                    decomped = _rank_3_tree(
                        wigner_3j_coefficients, l, L_R, M_R
                    )
                    coupling[M_R][rnk][lstr] = decomped
                elif rank == 4:
                    decomped = _rank_4_tree(
                        wigner_3j_coefficients, l, L_R, M_R
                    )
                    coupling[M_R][rnk][lstr] = decomped
                elif rank > 4:
                    raise ValueError(
                        "Cannot generate couplings for rank %d. Symmetric L_R "
                        "couplings have been implemented only up to rank 4."
                        % rank
                    )
    return coupling
