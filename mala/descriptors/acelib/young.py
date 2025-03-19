"""
Library of functions and classes for building Young diagrams and filling them.

This includes young subgroup fillings within orbits of irreproducible
representations of S_N.
"""

import math
import numpy as np

from mala.descriptors.acelib.common_utils import (
    flatten,
    group_vector_by_nodes,
    group_vector_by_orbits,
)
from mala.descriptors.acelib.tree_sorting import build_quick_tree
import itertools


class YoungSubgroup:
    """
    Class for Young tableau with subgroup filling options.

    Parameters
    ----------
    rank : int
        Rank of the Young tableau.

    Attributes
    ----------
    rank : int
        Rank of the Young tableau.

    partition : List
        Partition of the Young tableau.

    fills : List
        List of subgroup fillings.

    nodes : List or tuple
        List of nodes.

    remainder : any
        Remainder of building trees during filling of the subgroups.
    """

    def __init__(self, rank):
        self.rank = rank
        self.partition = None
        self.fills = None
        self.nodes = None
        self.remainder = None

    def sigma_c_partitions(self, max_orbit):
        """
        Get a list of partitions compatible with sigma_c.

        Parameters
        ----------
        max_orbit : int
            Maximum orbit size for permutation (P-cycle where P<N)

        Returns
        -------
        partitions : List
            List of partitions compatible with sigma_c.
        """
        # returns a list of partitions compatible with sigma_c
        nodes, remainder = build_quick_tree(range(self.rank))
        self.nodes = nodes
        self.remainder = remainder

        max_nc2 = math.floor(self.rank / 2)
        min_orbits = 1
        min_orbits += math.floor(self.rank / 4)

        min_nc1 = 0
        if remainder is not None:
            min_nc1 = 1
        if max_orbit is not None:
            max_orbits = max_orbit
        elif max_orbit is None:
            max_orbits = max_nc2 + min_nc1
        # orb_base = [i for i in range(1,self.rank+1) if i %2 ==0 or i == 1]
        orb_base = [i for i in range(1, self.rank + 1) if i == 2 or i == 1]
        if remainder != None:
            orb_base.append(1)
        global_sigma_c_parts = {}
        try:
            good_partitions = global_sigma_c_parts[self.rank]
        except KeyError:
            good_partitions = []
            for norbits in range(min_orbits, max_orbits + 1):
                possible_parts = [
                    tuple(self.reverse_vector(sorted(p)))
                    for p in itertools.product(orb_base, repeat=norbits)
                    if np.sum(p) == self.rank and max(p) <= 2**max_nc2
                ]
                good_partitions.extend(possible_parts)
            global_sigma_c_parts[self.rank] = list(set(good_partitions))
        return list(set(good_partitions))

    def check_single_fill(self, partition, inds, semistandard=True):
        """
        Check a that a collection of indices is sorted in a young tableau.

        This tableau will typically correspond to a young subgroup relevant for pairwise reduction of N spherical harmonics.

        Parameters
        ----------
        partition : List
            Partition of N determining shape of the Young tableau.

        inds : List
            Indices (filling) of the Young tableau.

        semistandard : bool
            Whether the Young tableau filling is semistandard.

        Returns
        -------
        bool
            True if the fill is correct, False otherwise.
        """
        tmpi = sorted(inds)
        unique_tmpi = list(set(tmpi))
        ii_to_inds = {}
        indi_to_ii = {}
        for ii, indi in zip(range(len(unique_tmpi)), unique_tmpi):
            ii_to_inds[ii] = indi
            indi_to_ii[indi] = ii

        place_holders = [indi_to_ii[ik] for ik in inds]
        if self.partition is None:
            self.partition = partition
        mapped_perm = tuple(place_holders)
        fill = group_vector_by_orbits(mapped_perm, partition)
        row_sort = self.is_row_sort(fill)
        col_sort = self.is_column_sort(fill)
        if semistandard:
            return row_sort and col_sort
        elif not semistandard:
            return row_sort

    def subgroup_fill(
        self,
        inds,
        partitions=None,
        max_orbit=None,
        semistandard=True,
        lreduce=False,
    ):
        """
        Fill the young subgroup subgroup according to standard, semistandard, etc conventions.
        
        This will generate multiple fillings relevant for the pairwise reduction of N spherical harmonics. 
        
        Parameters
        ----------
        inds : List
            Indices to be filled in the Young tableau.

        partitions : List
            Partitions of N determining the shape of the Young tableau.

        max_orbit : int
            Maximum orbit, aka max row size in the tableau.

        semistandard : bool
            Whether the Young tableau filling is semistandard.

        lreduce : bool
            Whether to reduce the Young tableau fillings according to 2-cycles (according to pairwise angular momentum coupling scheme)
        """
        if partitions is None:
            partitions = self.sigma_c_partitions(max_orbit)
        if len(set(inds)) == 1:
            partitions = [tuple([len(inds)])] + partitions
        subgroup_fills = []
        fills_perpart = {tuple(partition): [] for partition in partitions}
        part_perfill = {}
        all_perms = self.unique_permutations(inds)

        # UNUSED CODE, maybe we will need this laterm, so I am not deleting it
        # directly.

        # get the full automorphism group including any expansion due to
        # degeneracy
        # G_N = get_auto_part(inds,partitions[0],add_degen_autos=True,part_
        # only=False)
        # applied_perms = [tuple(Permutation(filled_perm(pi,len(inds)))(inds))
        # for pi in G_N]
        # collect a group of permutations \sigma \in S_N \notin G_N
        # idi = [tuple([ki]) for ki in range(len(inds))]
        # H_N = [tuple(idi) ]
        # for raw_perm in perms_raw:
        #    P = Permutation(raw_perm)
        #    cyc = P.full_cyclic_form
        #    cyc = tuple([tuple(k) for k in cyc])
        #    this_applied = P(inds)
        #    if tuple(this_applied) not in applied_perms:
        #        H_N.append(cyc)
        # not_equals  = [tuple(Permutation(filled_perm(pi,len(inds)))(inds))
        # for pi in H_N]
        # if len(not_equals) != 0:
        #    loopperms = not_equals.copy()
        # elif len(not_equals) == 0:
        loopperms = all_perms.copy()
        if lreduce:
            tmp = []
            nodes, remainder = build_quick_tree(inds)
            for loopperm in loopperms:
                grouped = group_vector_by_nodes(loopperm, nodes, remainder)
                if remainder != None:
                    srted = sorted(grouped[:-1])
                    srted.append(grouped[-1])
                    srted = tuple(srted)
                elif remainder == None:
                    srted = sorted(grouped)
                    srted = tuple(srted)
                if tuple(grouped) == srted:
                    tmp.append(loopperm)
            loopperms = tmp.copy()

        for partition in partitions:
            for fill in loopperms:
                # flag,subgroup_fillings = self.check_subgroup_fill(partition,fill,sigma_c_symmetric=sigma_c_symmetric,semistandard=semistandard)
                flag = self.check_single_fill(
                    partition, fill, semistandard=semistandard
                )
                if flag and fill not in subgroup_fills:
                    subgroup_fills.append(fill)
                fills_perpart[tuple(partition)].append(fill)
                try:
                    part_perfill[fill].append(tuple(partition))
                except KeyError:
                    part_perfill[fill] = [tuple(partition)]
        all_fills = subgroup_fills  # list(set(subgroup_fills))
        self.fills = sorted(all_fills)
        for sgf, pts in part_perfill.items():
            pts = list(set(pts))
            pts.sort(key=lambda x: x.count(2), reverse=True)
            pts.sort(
                key=lambda x: tuple([i % 2 == 0 for i in x]), reverse=True
            )
            pts.sort(key=lambda x: max(x), reverse=True)
            part_perfill[sgf] = pts

    @staticmethod
    def unique_permutations(vec):
        """
        Get the unique permutations of a vector.

        Parameters
        ----------
        vec : List
            Vector to get the permutations of.

        Returns
        -------
        unique_perms : List
            List of unique permutations.
        """
        all_perms = [p for p in itertools.permutations(vec)]
        return sorted(list(set(all_perms)))

    @staticmethod
    def is_column_sort(partitionfill, strict_col_sort=False):
        """
        Check if the young tableau is column sorted.

        Parameters
        ----------
        partitionfill : List
            Partition fill to check.

        strict_col_sort : bool
            Whether to check for strict column sorting in the young tableau. If True, 
            entries in columns of the tableau must be strictly increasing. 

        Returns
        -------
        sorted : bool
            True if the columns are sorted, False otherwise.
        """
        lens = [len(x) for x in partitionfill]
        ranges = [list(range(ln)) for ln in lens]
        cols = list(set(flatten(ranges)))
        bycol = {col: [] for col in cols}
        for subrange, orbitlst in zip(ranges, partitionfill):
            for colidx, orbitval in zip(subrange, orbitlst):
                bycol[colidx].append(orbitval)
        coltups = [tuple(bycol[colidx]) for colidx in cols]
        sortedcols = [tuple(sorted(bycol[colidx])) for colidx in cols]
        # check to see if columns are sorted
        sortedcol_flag = all([a == b for a, b in zip(coltups, sortedcols)])
        if strict_col_sort:
            sortedcol_flag = sortedcol_flag and all(
                [len(list(set(a))) == len(a) for a in coltups]
            )
        return sortedcol_flag

    @staticmethod
    def is_row_sort(partitionfill):
        """
        Check if the Young Tableau filling is row sorted.

        Parameters
        ----------
        partitionfill : List
            Partition fill to check.

        Returns
        -------
        sorted : bool
            True if the rows are sorted, False otherwise.
        """
        all_srt = []
        for orbit in partitionfill:
            logi = tuple(sorted(orbit)) == orbit
            all_srt.append(logi)
        return all(all_srt)

    @staticmethod
    def reverse_vector(v):
        """
        Reverse a vector.

        Returns a copy of the vector, to allow reverse and actual vector
        to both be present. This is just a wrapper for the python reverse() function.

        Parameters
        ----------
        v : List
            Vector to reverse.

        Returns
        -------
        v_reversed : List
            Reversed vector.
        """
        vtmp = v.copy()
        vtmp.reverse()
        return vtmp
