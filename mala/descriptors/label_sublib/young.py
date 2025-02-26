from sympy.combinatorics import Permutation

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

# library of functions and classes for building young diagrams and filling them
# this includes young subgroup fillings within orbits of irreps of S_N
global_sigma_c_parts = {}


def Reverse(v):
    vtmp = v.copy()
    vtmp.reverse()
    return vtmp


def flatten(lstoflsts):
    try:
        flat = [i for sublist in lstoflsts for i in sublist]
        return flat
    except TypeError:
        return lstoflsts


def filled_perm(tups, rank):
    allinds = list(range(rank))
    try:
        remainders = [i for i in allinds if i not in flatten(tups)]
        alltups = tups + tuple([tuple([k]) for k in remainders])
    except TypeError:
        remainders = [i for i in allinds if i not in flatten(flatten(tups))]
        alltups = tups + tuple([tuple([k]) for k in remainders])
    return Permutation(alltups)


def is_column_sort(partitionfill, strict_col_sort=False):
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


def is_row_sort(partitionfill):
    all_srt = []
    for orbit in partitionfill:
        logi = tuple(sorted(orbit)) == orbit
        all_srt.append(logi)
    return all(all_srt)


# TODO: Rename class.
class YoungSubgroup:
    # Class for young tableau with subgroup filling options

    def __init__(self, rank):
        self.rank = rank
        self.partition = None
        self.fills = None
        self.nodes = None
        self.remainder = None

    def sigma_c_partitions(self, max_orbit):
        # returns a list of partitions compatible with sigma_c
        nodes, remainder = tree(range(self.rank))
        self.nodes = nodes
        self.remainder = remainder

        max_nc2 = math.floor(self.rank / 2)
        min_orbits = 1
        min_orbits += math.floor(self.rank / 4)

        min_nc1 = 0
        if remainder != None:
            min_nc1 = 1
        if max_orbit != None:
            max_orbits = max_orbit
        elif max_orbit == None:
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
                    tuple(Reverse(sorted(p)))
                    for p in itertools.product(orb_base, repeat=norbits)
                    if np.sum(p) == self.rank and max(p) <= 2**max_nc2
                ]
                good_partitions.extend(possible_parts)
            global_sigma_c_parts[self.rank] = list(set(good_partitions))
        return list(set(good_partitions))

    def check_single_fill(self, partition, inds, semistandard=True):
        tmpi = sorted(inds)
        unique_tmpi = list(set(tmpi))
        ii_to_inds = {}
        indi_to_ii = {}
        for ii, indi in zip(range(len(unique_tmpi)), unique_tmpi):
            ii_to_inds[ii] = indi
            indi_to_ii[indi] = ii

        place_holders = [indi_to_ii[ik] for ik in inds]
        if self.partition == None:
            self.partition = partition
        mapped_perm = tuple(place_holders)
        fill = group_vec_by_orbits(mapped_perm, partition)
        row_sort = is_row_sort(fill)
        col_sort = is_column_sort(fill)
        if semistandard:
            return row_sort and col_sort
        elif not semistandard:
            return row_sort

    def subgroup_fill(
        self,
        inds,
        partitions=None,
        max_orbit=None,
        sigma_c_symmetric=False,
        semistandard=True,
        lreduce=False,
    ):
        if partitions == None:
            partitions = self.sigma_c_partitions(max_orbit)
        if len(set(inds)) == 1:
            partitions = [tuple([len(inds)])] + partitions
        subgroup_fills = []
        fills_perpart = {tuple(partition): [] for partition in partitions}
        part_perfill = {}
        all_perms = unique_perms(inds)

        # get the full automorphism group including any expansion due to degeneracy
        # G_N = get_auto_part(inds,partitions[0],add_degen_autos=True,part_only=False)
        # applied_perms = [tuple(Permutation(filled_perm(pi,len(inds)))(inds)) for pi in G_N]

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
        # not_equals  = [tuple(Permutation(filled_perm(pi,len(inds)))(inds)) for pi in H_N]
        # if len(not_equals) != 0:
        #    loopperms = not_equals.copy()
        # elif len(not_equals) == 0:
        loopperms = all_perms.copy()
        if lreduce:
            tmp = []
            nodes, remainder = tree(inds)
            for loopperm in loopperms:
                grouped = group_vec_by_node(loopperm, nodes, remainder)
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
