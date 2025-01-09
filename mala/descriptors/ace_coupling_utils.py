"""Utility functions for calculation of ACE coupling coefficients."""

import itertools

import numpy as np
from mala.descriptors.pa_lib import *
from mala.descriptors.pa_gen import *

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
            node: get_intermediates_w(l[node[0]], l[node[1]]) for node in nodes
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
                flag = check_triangle(ltup[0], ltup[1], L_R) and parity_flag
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


def pa_labels_raw(rank, nmax, lmax, mumax, lmin=1, L_R=0, M_R=0):
    if rank >= 4:
        all_max_l = 12
        all_max_n = 12
        all_max_mu = 8
        try:
            with open(
                "%s/all_labels_mu%d_n%d_l%d_r%d.json"
                % (lib_path, all_max_mu, all_max_n, all_max_l, rank),
                "r",
            ) as readjson:
                data = json.load(readjson)
        except FileNotFoundError:
            build_tabulated(rank, all_max_mu, all_max_n, all_max_l, L_R, M_R)
            with open(
                "%s/all_labels_mu%d_n%d_l%d_r%d.json"
                % (lib_path, all_max_mu, all_max_n, all_max_l, rank),
                "r",
            ) as readjson:
                data = json.load(readjson)
        lmax_strs = generate_l_LR(
            range(lmin, lmax + 1), rank, L_R=L_R, M_R=M_R
        )
        lvecs = [
            tuple([int(k) for k in lmax_str.split(",")])
            for lmax_str in lmax_strs
        ]
        # nvecs = [i for i in itertools.combinations_with_replacement(range(0,nmax),rank)]
        muvecs = [
            i
            for i in itertools.combinations_with_replacement(
                range(mumax), rank
            )
        ]
        # reduced_nvecs=get_mapped_subset(nvecs)

        all_lammps_labs = []
        all_not_compat = []
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
                nus = from_tabulated(
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
                all_not_compat.extend(not_compatible)

                # print ('raw PA-RPI',nus)
                # print ('lammps ready PA-RPI',lammps_ready)
                # print ('not compatible with lammps (PA-RPI with a nu vector that cannot be reused)',not_compatible)
    elif rank < 4:
        # no symmetry reduction required for rank <= 3
        # use typical lexicographical ordering for such cases
        labels = generate_nl(
            rank,
            nmax,
            lmax,
            mumax=mumax,
            lmin=lmin,
            L_R=L_R,
            M_R=M_R,
            all_perms=False,
        )
        all_lammps_labs = labels
        all_not_compat = []

    return all_lammps_labs, all_not_compat


def generate_nl(
    rank, nmax, lmax, mumax=1, lmin=0, L_R=0, M_R=0, all_perms=False
):
    # rank: int  - basis function rank to evaluate nl combinations for
    # nmax: int  - maximum value of the n quantum numbers in the nl vectors
    # lmax: int  - maximum value of the l quantum numbers in the nl vectors
    # mumax: int  - maximum value of the chemical variable in the munl vectors (default is none for single component system)
    # RETURN: list of munl vectors in string format mu0_mu1,mu2,...muk,n1,n2,..n_k,l1,l2,..l_k_L1-L2...-LK
    # NOTE: All valid intermediates L are generated

    munl = []

    murng = range(mumax)
    nrng = range(1, nmax + 1)
    lrng = range(lmin, lmax + 1)

    mus = ind_vec(murng, rank)
    ns = ind_vec(nrng, rank)
    ls = generate_l_LR(lrng, rank, L_R)

    linters_per_l = {
        l: tree_l_inters([int(b) for b in l.split(",")], L_R=0) for l in ls
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


def build_tabulated(rank, all_max_mu, all_max_n, all_max_l, L_R=0, M_R=0):
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
    muvecs = [
        i
        for i in itertools.combinations_with_replacement(
            range(all_max_mu), rank
        )
    ]
    reduced_nvecs = get_mapped_subset(nvecs)
    fs_labs = []
    all_nl = []

    all_PA_tabulated = []
    PA_per_nlblock = {}
    for nin in reduced_nvecs:
        for lin in lvecs:
            max_labs, all_labs, labels_per_block, original_spans = tree_labels(
                nin, lin
            )
            combined_labs = combine_blocks(
                labels_per_block, lin, original_spans
            )
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
            mustrlst = ["%d"] * rank
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
                mu0, mu, n, l, L = get_mu_n_l(lab, return_L=True)
                if L != None:
                    nlL = (tuple(n), tuple(l), L)
                else:
                    nlL = (tuple(n), tuple(l), tuple([]))
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
        "%s/all_labels_mu%d_n%d_l%d_r%d.json"
        % (lib_path, all_max_mu, all_max_n, all_max_l, rank),
        "w",
    ) as writejson:
        json.dump(dct, writejson, sort_keys=False, indent=2)


def from_tabulated(mu, n, l, allowed_mus=[0], tabulated_all=None):
    rank = len(l)
    Lveclst = ["%d"] * (rank - 2)
    vecstrlst = ["%d"] * rank
    unique_mun, mun_tupped = muvec_nvec_combined(mu, n)
    all_labels = []
    for mun_tup in mun_tupped:
        mappedn, mappedl, mprev_n, mprev = get_mapped(mun_tup, l)
        this_key = (tuple(mappedn), tuple(l))
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
    N = len(lin)
    uniques = list(set(lin))
    tmp = list(lin).copy()
    tmp.sort(key=Counter(lin).get, reverse=True)
    uniques.sort()
    uniques.sort(key=Counter(tmp).get, reverse=True)
    count_uniques = [lin.count(u) for u in uniques]
    mp = {uniques[i]: i for i in range(len(uniques))}
    mprev = {i: uniques[i] for i in range(len(uniques))}
    mappedl = [mp[t] for t in tmp]

    unique_ns = list(set(nin))
    tmpn = list(nin).copy()
    tmpn.sort(key=Counter(nin).get, reverse=True)
    unique_ns.sort()
    unique_ns.sort(key=Counter(nin).get, reverse=True)
    count_unique_ns = [nin.count(u) for u in unique_ns]
    mp_n = {unique_ns[i]: i for i in range(len(unique_ns))}
    mprev_n = {i: unique_ns[i] for i in range(len(unique_ns))}
    mappedn = [mp_n[t] for t in tmpn]
    mappedn = tuple(mappedn)
    mappedl = tuple(mappedl)
    return mappedn, mappedl, mprev_n, mprev


def muvec_nvec_combined(mu, n):
    mu = sorted(mu)
    # n = sorted(n)
    umus = sorted(list(set(itertools.permutations(mu))))
    uns = sorted(list(set(itertools.permutations(n))))
    combos = [cmb for cmb in itertools.product(umus, uns)]
    tupped = [tuple([(ni, mui) for mui, ni in zip(*cmb)]) for cmb in combos]
    tupped = [
        tuple(sorted([(ni, mui) for mui, ni in zip(*cmb)])) for cmb in combos
    ]
    tupped = list(set(tupped))
    uniques = []
    for tupi in tupped:
        nil = []
        muil = []
        for tupii in tupi:
            muil.append(tupii[1])
            nil.append(tupii[0])
        uniques.append(tuple([tuple(muil), tuple(nil)]))
    return uniques, tupped


def ind_vec(lrng, size):
    uniques = []
    combs = itertools.combinations_with_replacement(lrng, size)
    for comb in combs:
        perms = itertools.permutations(comb)
        for p in perms:
            pstr = ",".join(str(k) for k in p)
            if pstr not in uniques:
                uniques.append(pstr)
    return uniques


def get_mu_n_l(nu_in, return_L=False, **kwargs):
    rank = get_mu_nu_rank(nu_in)
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


def get_mu_nu_rank(nu_in):
    if len(nu_in.split("_")) > 1:
        assert (
            len(nu_in.split("_")) <= 3
        ), "make sure your descriptor label is in proper format: mu0_mu1,mu2,mu3,n1,n2,n3,l1,l2,l3_L1"
        nu = nu_in.split("_")[1]
        nu_splt = nu.split(",")
        return int(len(nu_splt) / 3)
    else:
        nu = nu_in
        nu_splt = nu.split(",")
        return int(len(nu_splt) / 2)
