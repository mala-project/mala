import itertools
import json

import numpy as np

import mala.descriptors.ace_coupling_utils as acu


class ACEPotential:
    def __init__(
        self,
        elements,
        reference_ens,
        ranks,
        nmax,
        lmax,
        nradbase,
        rcut,
        lmbda,
        css,
        rcutinner=0.0,
        drcutinner=0.01,
        lmin=1,
        b_basis="pa_tabulated",  #'pa_tabulated', 'minsub', 'ysg_x_so3'
        manuallabs=None,
        **kwargs
    ):
        if kwargs is not None:
            self.__dict__.update(kwargs)

        self.global_ccs = css
        self.global_ccs[1] = {"0": {tuple([]): {"0": 1.0}}}
        self.E0 = reference_ens
        self.ranks = ranks
        self.elements = elements
        self.betas = None
        self.nus = None
        self.deltaSplineBins = 0.001
        self.global_ndensity = 1
        self.global_rhocut = 100000
        self.global_drhocut = 250

        # assert the same nmax,lmax,nradbase (e.g. same basis) for each bond type
        self.radbasetype = "ChebExpCos"
        self.global_nmax = nmax
        self.global_lmax = lmax
        self.b_basis = b_basis
        assert len(nmax) == len(lmax), "nmax and lmax arrays must be same size"

        self.global_nradbase = nradbase

        # These can be global or per bond type, global_mode controls which
        # of these settings is used.
        self.rcut = rcut
        self.lmbda = lmbda
        self.rcutinner = rcutinner
        self.drcutinner = drcutinner
        self.lmin = lmin
        self.global_mode = False

        if not isinstance(self.rcut, dict) and not isinstance(self.rcut, list):
            self.global_mode = True

        print(self.global_mode)

        self.manuallabs = manuallabs
        self.set_embeddings()
        self.set_bonds()
        self.set_bond_base()

        lmax_dict = {
            rank: lv for rank, lv in zip(self.ranks, self.global_lmax)
        }
        try:
            lmin_dict = {rank: lv for rank, lv in zip(self.ranks, self.lmin)}
        except AttributeError:
            lmin_dict = {
                rank: lv
                for rank, lv in zip(
                    self.ranks, self.global_lmin * len(self.ranks)
                )
            }
        nradmax_dict = {
            rank: nv for rank, nv in zip(self.ranks, self.global_nmax)
        }
        mumax_dict = {rank: len(self.elements) for rank in self.ranks}

        if self.manuallabs is not None:
            with open(self.manuallabs, "r") as readjson:
                labdata = json.load(readjson)
            nulst_1 = [list(ik) for ik in list(labdata.values())]
        # If I am not mistaken, then this option is currently incomplete.
        # I have commented it out, while also for now removing the option
        # that relied on FitSNAP, because we do not want to ship MALA with
        # FitSNAP as a requirement at the moment.
        # elif self.b_basis == "ysg_x_so3":
        #
        #     nulst_1 = []
        else:
            nulst_1 = []
            for rank in self.ranks:
                PA_lammps = acu.pa_labels_raw(
                    rank,
                    nradmax_dict[rank],
                    lmax_dict[rank],
                    mumax_dict[rank],
                    lmin_dict[rank],
                )
                nulst_1.append(PA_lammps)

        nus_unsort = [item for sublist in nulst_1 for item in sublist]
        nus = nus_unsort.copy()
        mu0s = []
        mus = []
        ns = []
        ls = []
        for nu in nus_unsort:
            mu0ii, muii, nii, lii = acu.get_mu_n_l(nu)
            mu0s.append(mu0ii)
            mus.append(tuple(muii))
            ns.append(tuple(nii))
            ls.append(tuple(lii))
        nus.sort(key=lambda x: mus[nus_unsort.index(x)], reverse=False)
        nus.sort(key=lambda x: ns[nus_unsort.index(x)], reverse=False)
        nus.sort(key=lambda x: ls[nus_unsort.index(x)], reverse=False)
        nus.sort(key=lambda x: mu0s[nus_unsort.index(x)], reverse=False)
        nus.sort(key=lambda x: len(x), reverse=False)
        nus.sort(key=lambda x: mu0s[nus_unsort.index(x)], reverse=False)
        self.nus = nus
        self.set_funcs(nus)

    def set_embeddings(
        self, npoti="FinnisSinclair", FSparams=[1.0, 1.0]
    ):  # default for linear models in lammps PACE
        # embeddings =dict()#OrderedDict() #{ind:None for ind in range(len(self.elements))}
        embeddings = {ind: None for ind in range(len(self.elements))}
        for elemind in range(len(self.elements)):
            embeddings[elemind] = {
                "ndensity": self.global_ndensity,
                "FS_parameters": FSparams,
                "npoti": npoti,
                "rho_core_cutoff": self.global_rhocut,
                "drho_core_cutoff": self.global_drhocut,
            }
        self.embeddings = embeddings

    def set_bonds(self):
        bondinds = range(len(self.elements))
        bond_lsts = [list(b) for b in itertools.product(bondinds, bondinds)]
        self.bondlsts = bond_lsts

    def set_bond_base(self):
        bondstrs = ["[%d, %d]" % (b[0], b[1]) for b in self.bondlsts]
        bonds = {bondstr: None for bondstr in bondstrs}

        # radial basis function expansion coefficients
        # saved in n,l,k shape
        # defaults to orthogonal delta function [g(n,k)] basis of drautz 2019
        try:
            nradmax = max(self.global_nmax[:])
        except ValueError:
            nradmax = max(self.global_nmax)
        lmax = max(self.global_lmax)
        nradbase = self.global_nradbase
        crad = np.zeros((nradmax, lmax + 1, nradbase), dtype=int)
        for n in range(nradmax):
            for l in range(lmax + 1):
                crad[n][l] = np.array(
                    [1 if k == n else 0 for k in range(nradbase)]
                )

        cnew = np.zeros((nradbase, nradmax, lmax + 1))
        for n in range(1, nradmax + 1):
            for l in range(lmax + 1):
                for k in range(1, nradbase + 1):
                    cnew[k - 1][n - 1][l] = crad[n - 1][l][k - 1]

        for bondind, bondlst in enumerate(self.bondlsts):
            bstr = "[%d, %d]" % (bondlst[0], bondlst[1])
            # TODO: Is there a way to assign this without try/except, but
            #  rather an if-condition?
            if self.global_mode:
                bonds[bstr] = {
                    "nradmax": nradmax,
                    "lmax": max(self.global_lmax),
                    "nradbasemax": self.global_nradbase,
                    "radbasename": self.radbasetype,
                    "radparameters": [self.global_lmbda],
                    "radcoefficients": crad.tolist(),
                    "prehc": 0,
                    "lambdahc": self.lmbda,
                    "rcut": self.rcut,
                    "dcut": 0.01,
                    "rcut_in": self.rcutinner,
                    "dcut_in": self.drcutinner,
                    "inner_cutoff_type": "distance",
                }
            else:
                if isinstance(self.rcut, dict):
                    bonds[bstr] = {
                        "nradmax": nradmax,
                        "lmax": max(self.global_lmax),
                        "nradbasemax": self.global_nradbase,
                        "radbasename": self.radbasetype,
                        "radparameters": [self.lmbda[bstr]],
                        "radcoefficients": crad.tolist(),
                        "prehc": 0,
                        "lambdahc": self.lmbda[bstr],
                        "rcut": self.rcut[bstr],
                        "dcut": 0.01,
                        "rcut_in": self.rcutinner[bstr],
                        "dcut_in": self.drcutinner[bstr],
                        "inner_cutoff_type": "distance",
                    }
                elif isinstance(self.rcut, list):
                    bonds[bstr] = {
                        "nradmax": nradmax,
                        "lmax": max(self.global_lmax),
                        "nradbasemax": self.global_nradbase,
                        "radbasename": self.radbasetype,
                        "radparameters": [self.lmbda[bondind]],
                        "radcoefficients": crad.tolist(),
                        "prehc": 0,
                        "lambdahc": self.lmbda[bondind],
                        "rcut": self.rcut[bondind],
                        "dcut": 0.01,
                        "rcut_in": self.rcutinner[bondind],
                        "dcut_in": self.drcutinner[bondind],
                        "inner_cutoff_type": "distance",
                    }

        self.bonds = bonds

    def set_funcs(self, nulst=None, muflg=True, print_0s=True):

        # TODO: Simplify this.
        if nulst == None:
            if self.nus != None:
                nulst = self.nus.copy()
            else:
                raise AttributeError("No list of descriptors found/specified")
        nus_per_rank = {}
        permu0 = {b: [] for b in range(len(self.elements))}
        permunu = {b: [] for b in range(len(self.elements))}
        if self.betas != None:
            betas = self.betas
        else:
            # betas = {ind:{nu:1.0 for nu in nulst} for ind in range(len(self.elements))}
            betas = {ind: {} for ind in range(len(self.elements))}
            for nu in nulst:
                mu0, mu, n, l, L = acu.get_mu_n_l(nu, return_L=True)
                betas[mu0][nu] = 1.0
        for nu in nulst:
            mu0, mu, n, l, L = acu.get_mu_n_l(nu, return_L=True)
            rank = acu.get_mu_nu_rank(nu)
            try:
                nus_per_rank[rank].append(nu)
            except KeyError:
                nus_per_rank[rank] = [nu]
            llst = ["%d"] * rank
            # print (nu,l,oldfmt,muflg)
            lstr = ",".join(b for b in llst) % tuple(l)
            if L != None:
                ccs = self.global_ccs[rank][lstr][tuple(L)]
            elif L == None:
                try:
                    ccs = self.global_ccs[rank][lstr][()]
                except KeyError:
                    ccs = self.global_ccs[rank][lstr]
            ms = list(ccs.keys())
            mslsts = [[int(k) for k in m.split(",")] for m in ms]
            msflat = [item for sublist in mslsts for item in sublist]
            if print_0s or betas[mu0][nu] != 0.0:
                ccoeffs = list(np.array(list(ccs.values())) * betas[mu0][nu])
                permu0[mu0].append(
                    {
                        "mu0": mu0,
                        "rank": rank,
                        "ndensity": self.global_ndensity,
                        "num_ms_combs": len(ms),
                        "mus": mu,
                        "ns": n,
                        "ls": l,
                        "ms_combs": msflat,
                        "ctildes": ccoeffs,
                    }
                )
                permunu[mu0].append(nu)
            elif betas[mu0][nu] == 0.0 and not print_0s:
                print("Not printing descriptor: %s, coefficient is 0" % nu)
        self.nus_per_rank = nus_per_rank

        # for b in range(len(self.elements)):
        #   for i in permunu[b]:
        #       print (b,i)
        # for b in range(len(self.elements)):
        #   print (b,len(permu0[b]))
        self.funcs = permu0
        self.permunu = permunu

    def write_pot(self, name):
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)

        with open("%s.yace" % name, "w") as writeout:
            e0lst = ["%f"] * len(self.elements)
            e0str = ", ".join(b for b in e0lst) % tuple(self.E0)
            elemlst = ["%s"] * len(self.elements)
            elemstr = ", ".join(b for b in elemlst) % tuple(self.elements)
            writeout.write("elements: [%s] \n" % elemstr)
            writeout.write("E0: [%s] \n" % e0str)
            writeout.write("deltaSplineBins: %f \n" % self.deltaSplineBins)
            writeout.write("embeddings:\n")
            for mu0, embed in self.embeddings.items():
                writeout.write("  %d: " % mu0)
                ystr = json.dumps(embed) + "\n"
                ystr = ystr.replace('"', "")
                writeout.write(ystr)
            writeout.write("bonds:\n")
            bondstrs = ["[%d, %d]" % (b[0], b[1]) for b in self.bondlsts]
            for bondstr in bondstrs:
                writeout.write("  %s: " % bondstr)
                bstr = json.dumps(self.bonds[bondstr]) + "\n"
                bstr = bstr.replace('"', "")
                writeout.write(bstr)
            writeout.write("functions:\n")
            for mu0 in range(len(self.elements)):
                writeout.write("  %d:\n" % (mu0))
                mufuncs = self.funcs[mu0]
                for mufunc in mufuncs:
                    mufuncstr = (
                        "    - " + json.dumps(mufunc, cls=NpEncoder) + "\n"
                    )
                    mufuncstr = mufuncstr.replace('"', "")
                    writeout.write(mufuncstr)

        return "%s.yace" % name
