"""Clebsch-Gordan coupling functions."""

import mala.descriptors.acelib.common_utils
from mala.descriptors.acelib import coupling_utils as ace_coupling_utils


def clebsch_gordan_coupling(
    clebsch_gordan_coefficients, ldict, L_R=0, use_permutations=True
):
    """
    Compute generalized Clebsch-Gordan coefficients for a given L_R.

    These are all defined according to the pairwise coupling scheme in Goff et al. 2024. 

    The approach to construct generalized Clebsch-Gordan coefficients is manually defined for each rank for pedagogical and clarity purposes. 

    Parameters
    ----------
    clebsch_gordan_coefficients : dict
        Precomputed Clebsch-Gordan coefficients.

    ldict : dict
        Dictionary of ranks and their corresponding l values.

    L_R : int
        Resultant angular momentum quantum number. This determines the equivariant
        character of the rank N descriptor after reduction. L_R=0 corresponds to
        a rotationally invariant feature, L_R=1 corresponds to a feature that
        transforms like a vector, L_R=2 a tensor, etc.

    use_permutations : bool
        Flag to consider permutations of l in generation of coupling coefficients.

    Returns
    -------
    coupling : dict
        Generalized Clebsch-Gordan coefficients, {M_R:{rank:{l:{m:coupling_coefficient}}}
    """
    # Define the functions for the individual ranks.
    # between the functions for the individual ranks are.

    def _rank_1_tree(_clebsch_gordan_coefficients, _l, _L_R=0, _M_R=0):
        """
        Compute the coupling for rank 1.

        Parameters
        ----------
        _clebsch_gordan_coefficients : dict
            Precomputed Clebsch-Gordan coefficients.

        _l : List
            list of angular momentum quantum numbers [l1]

        _L_R : int
            Resultant angular momentum quantum number. This determines the equivariant character of the rank N descriptor after reduction. L_R=0 corresponds to a rotationally invariant feature, L_R=1 corresponds to a feature that transforms like a vector, L_R=2 a tensor, etc.

        _M_R : int
            Resultant projection quantum number. This also determines the equivariant character of the rank N descriptor after reduction. M_R must obey -L_R <= M_R <= L_R.

        Returns
        -------
        decomposed : dict
            Generalized Clebsch-Gordan coefficient for rank 1
        """
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
                conds = m_ints[0] == _M_R
                if conds:
                    w1 = _clebsch_gordan_coefficients[
                        "%d,%d,%d,%d,%d,%d"
                        % (_l[0], m_ints[0], _L_R, _M_R, 0, 0)
                    ]
                    phase = 1
                    w = phase * w1

                    decomposed[inter][mstr] = w
        return decomposed

    def _rank_2_tree(_clebsch_gordan_coefficients, _l, _L_R=0, _M_R=0):
        """
        Compute the coupling for rank 2.

        Parameters
        ----------
        _clebsch_gordan_coefficients : dict
            Precomputed Clebsch-Gordan coefficients.

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
            Generalized Clebsch-Gordan coefficients for rank 2

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
                    w1 = _clebsch_gordan_coefficients[
                        "%d,%d,%d,%d,%d,%d"
                        % (_l[0], m_ints[0], _l[1], m_ints[1], _L_R, _M_R)
                    ]
                    phase = 1
                    w = phase * w1

                    decomposed[inter][mstr] = w
        return decomposed

    def _rank_3_tree(_clebsch_gordan_coefficients, _l, _L_R=0, _M_R=0):
        """
        Compute the coupling for rank 3.

        Parameters
        ----------
        _clebsch_gordan_coefficients : dict
            Precomputed Clebsch-Gordan coefficients.

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
            Generalized Clebsch-Gordan coefficients for rank 3
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
                        w1 = _clebsch_gordan_coefficients[
                            "%d,%d,%d,%d,%d,%d"
                            % (_l[0], m_ints[0], _l[1], m_ints[1], L1, M1)
                        ]
                        w2 = _clebsch_gordan_coefficients[
                            "%d,%d,%d,%d,%d,%d"
                            % (L1, M1, _l[2], m_ints[2], _L_R, _M_R)
                        ]
                        phase = 1
                        w = phase * w1 * w2
                        decomposed[inter][mstr] = w
        return decomposed

    def _rank_4_tree(_clebsch_gordan_coefficients, _l, _L_R=0, _M_R=0):
        """
        Compute the coupling for rank 4.

        Parameters
        ----------
        _clebsch_gordan_coefficients : dict
            Precomputed Clebsch-Gordan coefficients.

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
            Generalized Clebsch-Gordan coefficients for rank 4
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
                            w1 = _clebsch_gordan_coefficients[
                                "%d,%d,%d,%d,%d,%d"
                                % (_l[0], m_ints[0], _l[1], m_ints[1], L1, M1)
                            ]
                            w2 = _clebsch_gordan_coefficients[
                                "%d,%d,%d,%d,%d,%d"
                                % (_l[2], m_ints[2], _l[3], m_ints[3], L2, M2)
                            ]
                            w3 = _clebsch_gordan_coefficients[
                                "%d,%d,%d,%d,%d,%d"
                                % (L1, M1, L2, M2, _L_R, _M_R)
                            ]
                            # phase_power = ( L1 + L2 ) - ( M1 + M2 )  + ( L_R - M_R)
                            # phase = (-1) ** phase_power
                            phase = 1
                            w = phase * w1 * w2 * w3

                            decomposed[inter][mstr] = w
        return decomposed

    M_Rs = list(range(-L_R, L_R + 1))
    # generic coupling for any L_R - support must be added to call
    ranks = list(ldict.keys())
    coupling = {M_R: {rank: {} for rank in ranks} for M_R in M_Rs}

    #NOTE higher-rank couplings (to enable higher rank descriptors)
    # are available in FitSNAP
    for M_R in M_Rs:
        for rank in ranks:
            rnk = rank
            ls_per_rnk = ace_coupling_utils.generate_l_LR(
                range(ldict[rank] + 1),
                rank,
                L_R,
                M_R,
                use_permutations=use_permutations,
            )
            for lstr in ls_per_rnk:
                l = [int(k) for k in lstr.split(",")]
                if rank == 1:
                    decomped = _rank_1_tree(
                        clebsch_gordan_coefficients, l, L_R, M_R
                    )
                    coupling[M_R][rnk][lstr] = decomped
                elif rank == 2:
                    decomped = _rank_2_tree(
                        clebsch_gordan_coefficients, l, L_R, M_R
                    )
                    coupling[M_R][rnk][lstr] = decomped
                elif rank == 3:
                    decomped = _rank_3_tree(
                        clebsch_gordan_coefficients, l, L_R, M_R
                    )
                    coupling[M_R][rnk][lstr] = decomped
                elif rank == 4:
                    decomped = _rank_4_tree(
                        clebsch_gordan_coefficients, l, L_R, M_R
                    )
                    coupling[M_R][rnk][lstr] = decomped
                elif rank > 4:
                    raise ValueError(
                        "Cannot generate couplings for rank %d. symmetric L_R "
                        "couplings up to rank 4 have been implemented" % rank
                    )
    return coupling
