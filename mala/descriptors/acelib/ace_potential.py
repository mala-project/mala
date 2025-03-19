"""
ACE potential class.

It writes a .yace file (ACE potential file) that is readable by LAMMPS. When only the
coupling coefficients are saved to the .yace file, this acts as a parameter
file with which one can evaluate ACE descriptors.
"""

import itertools
import json

import numpy as np

import mala.descriptors.acelib.coupling_utils as ace_coupling_utils


class ACEPotential:
    """
    Class to manage interface between ACE descriptor enumeration library and LAMMPS. By default, it will enumerate ACE descriptors according to the Permutation-adapted approach described in https://doi.org/10.1016/j.jcp.2024.113073.

    However, there are options for assigning descriptor labels if desired, manually for example.

    After enumerating descriptors, it assigns all relevant hyperparamters needed to evaluate the ACE descriptors in LAMMPS. It saves a .yace file needed to evaluate ACE descriptors in LAMMPS (containing coupling coefficients). It does this by writing an ACE potential file, readable by LAMNMPS, that contains only information to evaluate descriptors, not energy models.
    
    Parameters
    ----------
    elements : List
        List of elements (symbols), including 'G' for the grid points. In ACE, all possible
        combinations of these elements determine the "bond types" spanned by the ACE chemical
        basis (the chemical basis is the delta function basis used in Drautz 2019). For
        example, the "bond types" resulting from all possible combinations of ['Al','G'] are
        determined with :code:`itertools.product()` in python. They are (Al,Al)(Al,G)(G,Al)(G,G).
        For mala, only those of type (G,X) are kept, (grid-atom interactions) and only a placeholder
        is kept for other interaction types.

    reference_ens : List
        List of reference energies. (To be applied only for linear models) with a constant
        shift to the energy per element type. Values other than 0 not be necessary in MALA.

    ranks : List
        Orders of the expansion, referred to as `N` in Drautz 2019, of the
        descriptors to be enumerated

    nmax: List
        Maximum radial basis function index per descriptor rank

    lmax: List
        Maximum angular momentum number per descriptor rank (maximum angular function index)

    nradbase : int
        Maximum radial basis function index OVERALL: max(nmax) - in the future, may be used
        to define the number of g_k(r) comprising R_nl from Drautz 2019 radial basis.

    rcut : float/list
        radial basis function cutoff per bond type. For example, if elements are ['Al','G']
        then rcut must be supplied for each:(Al,Al)(Al,G)(G,Al)(G,G)

    lmbda : float/list
        Exponential factor for scaled distance in Drautz 2019 used in the radial basis
        functions. As with the radial cutoff, lambda must be supplied per bond type. For
        example, if elements are ['Al','G'] then lambda must be supplied for 
        each:(Al,Al)(Al,G)(G,Al)(G,G)

    css : dict
        Dictionary of coupling coefficients of the format: {rank:{l:{m:coupling_coefficient}}

    rcutinner : List
        Inner cutoff to turn on soft-core repulsions. This parameter should be 0.0 (OFF) for
        each bond type in MALA. 

    drcutinner : List
        Parameter for soft-core repulsions. This parameter should not matter if rcutinner is
         0.0 (OFF) for each bond type in MALA. 

    lmin : int/list
        Lower bound on angular momentum quantum number per rank.

    manual_labels : str
        File for loading labels. If not None, then labels will be loaded from
        this json file. If None, then labels will be generated using the
        pa_labels_raw function (default).

    **kwarg : dict
        Additional keyword arguments.

    Returns
    -------
    ACEPotential : class
        Class containing ACE descriptor and hyperparamter info.
    """

    def __init__(
        self,
        elements,
        reference_energy,
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
        manual_labels=None,
        **kwargs
    ):
        if kwargs is not None:
            self.__dict__.update(kwargs)

        #NOTE Our unused variable names here match exactly what is in LAMMPS
        # I'm hesitant to change these to something else. We wont use them in MALA
        # and if people are really interested, they can find them in the lammps
        # source code directly
        #coupling coefficients (generalized Wigner symbols or Generalized Clebsch-Gordan coefficients)
        self.__global_ccs = css
        self.__global_ccs[1] = {"0": {tuple([]): {"0": 1.0}}}
        #0th-order expansion term in ACE (e.g., constant energy shift)
        self.__E0 = reference_energy
        self.__ranks = ranks
        self.__elements = elements
        #linear model coefficients
        self.__betas = None
        #descriptor labels in FitSNAP/LAMMPS format - as described elsewhere
        self.__nus = None
        #hyperparameters for ACE basis (relevant for density embeddings)
        self.__deltaSplineBins = 0.001
        self.__global_ndensity = 1
        self.__global_rhocut = 100000
        self.__global_drhocut = 250

        # assert the same nmax,lmax,nradbase (e.g. same basis) for each bond
        # type
        self.__radbasetype = "ChebExpCos"
        self.__global_nmax = nmax
        self.__global_lmax = lmax
        assert len(nmax) == len(lmax), "nmax and lmax arrays must be same size"

        self.__global_nradbase = nradbase

        # These can be global or per bond type, global_mode controls which
        # of these settings is used.
        self.__rcut = rcut
        self.__lmbda = lmbda
        self.__rcutinner = rcutinner
        self.__drcutinner = drcutinner
        self.__lmin = lmin
        self.__global_mode = False

        if not isinstance(self.__rcut, dict) and not isinstance(
            self.__rcut, list
        ):
            self.__global_mode = True

        self.__manual_labels = manual_labels
        self.__bondlsts = None
        self.__embeddings = None
        self.__bonds = None
        self.__set_embeddings()
        self.__set_bonds()
        self.__set_bond_base()

        lmax_dict = {
            rank: lv for rank, lv in zip(self.__ranks, self.__global_lmax)
        }
        try:
            lmin_dict = {
                rank: lv for rank, lv in zip(self.__ranks, self.__lmin)
            }
        except AttributeError:
            lmin_dict = {
                rank: lv
                for rank, lv in zip(
                    self.__ranks, self.global_lmin * len(self.__ranks)
                )
            }
        nradmax_dict = {
            rank: nv for rank, nv in zip(self.__ranks, self.__global_nmax)
        }
        mumax_dict = {rank: len(self.__elements) for rank in self.__ranks}

        if self.__manual_labels is not None:
            with open(self.__manual_labels, "r") as readjson:
                labdata = json.load(readjson)
            nulst_1 = [list(ik) for ik in list(labdata.values())]
        else:
            nulst_1 = []
            for rank in self.__ranks:
                PA_lammps = ace_coupling_utils.compute_pa_labels_raw(
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
            mu0ii, muii, nii, lii = ace_coupling_utils.calculate_mu_n_l(nu)
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
        self.__nus = nus
        self.set_funcs(nus)
        self.__nus_per_rank = None
        self.__funcs = None
        self.__permunu = None

    def __set_embeddings(self, npoti="FinnisSinclair", FSparams=[1.0, 1.0]):
        """
        Set 'embeddings' in the .yace file for lammps. Some terms here control if a square-root embedding term is added.

        This should always be OFF for descriptor calculations in MALA.

        Parameters
        ----------
        npoti : str
            paramter to specify descriptor type in LAMMPS

        FSparams : List
            parameters to specify embedding terms in LAMMPS
        """
        # embeddings =dict()#OrderedDict() #{ind:None for ind in range(len(self.elements))}
        embeddings = {ind: None for ind in range(len(self.__elements))}
        for elemind in range(len(self.__elements)):
            embeddings[elemind] = {
                "ndensity": self.__global_ndensity,
                "FS_parameters": FSparams,
                "npoti": npoti,
                "rho_core_cutoff": self.__global_rhocut,
                "drho_core_cutoff": self.__global_drhocut,
            }
        self.__embeddings = embeddings

    def __set_bonds(self):
        """Define a list of bonds, as given by element list."""
        bondinds = range(len(self.__elements))
        bond_lsts = [list(b) for b in itertools.product(bondinds, bondinds)]
        self.__bondlsts = bond_lsts

    def __set_bond_base(self):
        """
        Set the per-bond radial basis parameters. The 'bonds' are determined based on the elements.

        If the elements are ['Al','G'], the bonds may be determined with :code:`itertools.product()` in python. They are (Al,Al)(Al,G)(G,Al)(G,G).

        Some hyperparameters must be specified per bond label shown here.
        
        """
        bondstrs = ["[%d, %d]" % (b[0], b[1]) for b in self.__bondlsts]
        bonds = {bondstr: None for bondstr in bondstrs}

        # radial basis function expansion coefficients
        # saved in n,l,k shape
        # defaults to orthogonal delta function [g(n,k)] basis of drautz 2019
        try:
            nradmax = max(self.__global_nmax[:])
        except ValueError:
            nradmax = max(self.__global_nmax)
        lmax = max(self.__global_lmax)
        nradbase = self.__global_nradbase
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

        for bondind, bondlst in enumerate(self.__bondlsts):
            bstr = "[%d, %d]" % (bondlst[0], bondlst[1])
            if self.__global_mode:
                bonds[bstr] = {
                    "nradmax": nradmax,
                    "lmax": max(self.__global_lmax),
                    "nradbasemax": self.__global_nradbase,
                    "radbasename": self.__radbasetype,
                    "radparameters": [self.global_lmbda],
                    "radcoefficients": crad.tolist(),
                    "prehc": 0,
                    "lambdahc": self.__lmbda,
                    "rcut": self.__rcut,
                    "dcut": 0.01,
                    "rcut_in": self.__rcutinner,
                    "dcut_in": self.__drcutinner,
                    "inner_cutoff_type": "distance",
                }
            else:
                if isinstance(self.__rcut, dict):
                    bonds[bstr] = {
                        "nradmax": nradmax,
                        "lmax": max(self.__global_lmax),
                        "nradbasemax": self.__global_nradbase,
                        "radbasename": self.__radbasetype,
                        "radparameters": [self.__lmbda[bstr]],
                        "radcoefficients": crad.tolist(),
                        "prehc": 0,
                        "lambdahc": self.__lmbda[bstr],
                        "rcut": self.__rcut[bstr],
                        "dcut": 0.01,
                        "rcut_in": self.__rcutinner[bstr],
                        "dcut_in": self.__drcutinner[bstr],
                        "inner_cutoff_type": "distance",
                    }
                elif isinstance(self.__rcut, list):
                    bonds[bstr] = {
                        "nradmax": nradmax,
                        "lmax": max(self.__global_lmax),
                        "nradbasemax": self.__global_nradbase,
                        "radbasename": self.__radbasetype,
                        "radparameters": [self.__lmbda[bondind]],
                        "radcoefficients": crad.tolist(),
                        "prehc": 0,
                        "lambdahc": self.__lmbda[bondind],
                        "rcut": self.__rcut[bondind],
                        "dcut": 0.01,
                        "rcut_in": self.__rcutinner[bondind],
                        "dcut_in": self.__drcutinner[bondind],
                        "inner_cutoff_type": "distance",
                    }

        self.__bonds = bonds

    def set_funcs(self, nulst=None, print_0s=True):
        """
        Set ctilde 'functions' in the .yace file, used to calculate descriptors in lammps.

        Parameters
        ----------
        nulst : List
            List of nus, a.k.a. ACE descriptor labels to write to 
            the .yace file.

        print_0s : bool
            Logical to include 0-valued descriptors, this should always
            be True in MALA as 0-valued descriptors only arise after 
            training linear models with sparsifying solvers like LASSO
        """
        if nulst is None:
            if self.__nus is not None:
                nulst = self.__nus.copy()
            else:
                raise AttributeError("No list of descriptors found/specified")
        nus_per_rank = {}
        permu0 = {b: [] for b in range(len(self.__elements))}
        permunu = {b: [] for b in range(len(self.__elements))}
        if self.__betas != None:
            betas = self.__betas
        else:
            # betas = {ind:{nu:1.0 for nu in nulst} for ind in range(len(self.elements))}
            betas = {ind: {} for ind in range(len(self.__elements))}
            for nu in nulst:
                mu0, mu, n, l, L = ace_coupling_utils.calculate_mu_n_l(
                    nu, return_L=True
                )
                betas[mu0][nu] = 1.0
        for nu in nulst:
            mu0, mu, n, l, L = ace_coupling_utils.calculate_mu_n_l(
                nu, return_L=True
            )
            rank = ace_coupling_utils.calculate_mu_nu_rank(nu)
            try:
                nus_per_rank[rank].append(nu)
            except KeyError:
                nus_per_rank[rank] = [nu]
            llst = ["%d"] * rank
            # print (nu,l,oldfmt,muflg)
            lstr = ",".join(b for b in llst) % tuple(l)
            if L != None:
                ccs = self.__global_ccs[rank][lstr][tuple(L)]
            elif L == None:
                try:
                    ccs = self.__global_ccs[rank][lstr][()]
                except KeyError:
                    ccs = self.__global_ccs[rank][lstr]
            ms = list(ccs.keys())
            mslsts = [[int(k) for k in m.split(",")] for m in ms]
            msflat = [item for sublist in mslsts for item in sublist]
            if print_0s or betas[mu0][nu] != 0.0:
                ccoeffs = list(np.array(list(ccs.values())) * betas[mu0][nu])
                permu0[mu0].append(
                    {
                        "mu0": mu0,
                        "rank": rank,
                        "ndensity": self.__global_ndensity,
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
        self.__nus_per_rank = nus_per_rank

        # for b in range(len(self.elements)):
        #   for i in permunu[b]:
        #       print (b,i)
        # for b in range(len(self.elements)):
        #   print (b,len(permu0[b]))
        self.__funcs = permu0
        self.__permunu = permunu

    def write_pot(self, name):
        """
        Write coupling coefficients to file.

        Parameters
        ----------
        name : str
            Name (without file ending) of the file to write to.

        Returns
        -------
        name : str
            Filename of the written file (with file ending)
        """

        class _NPEncoder(json.JSONEncoder):
            """Helper class for encoding numpy arrays."""

            def default(self, obj):
                """
                Provdide conversion for numpy default types.

                Parameters
                ----------
                obj : any
                    Object to convert.

                Returns
                -------
                converted_obj : any
                    Converted object.
                """
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(_NPEncoder, self).default(obj)

        with open("%s.yace" % name, "w") as writeout:
            e0lst = ["%f"] * len(self.__elements)
            e0str = ", ".join(b for b in e0lst) % tuple(self.__E0)
            elemlst = ["%s"] * len(self.__elements)
            elemstr = ", ".join(b for b in elemlst) % tuple(self.__elements)
            writeout.write("elements: [%s] \n" % elemstr)
            writeout.write("E0: [%s] \n" % e0str)
            writeout.write("deltaSplineBins: %f \n" % self.__deltaSplineBins)
            writeout.write("embeddings:\n")
            for mu0, embed in self.__embeddings.items():
                writeout.write("  %d: " % mu0)
                ystr = json.dumps(embed) + "\n"
                ystr = ystr.replace('"', "")
                writeout.write(ystr)
            writeout.write("bonds:\n")
            bondstrs = ["[%d, %d]" % (b[0], b[1]) for b in self.__bondlsts]
            for bondstr in bondstrs:
                writeout.write("  %s: " % bondstr)
                bstr = json.dumps(self.__bonds[bondstr]) + "\n"
                bstr = bstr.replace('"', "")
                writeout.write(bstr)
            writeout.write("functions:\n")
            for mu0 in range(len(self.__elements)):
                writeout.write("  %d:\n" % (mu0))
                mufuncs = self.__funcs[mu0]
                for mufunc in mufuncs:
                    mufuncstr = (
                        "    - " + json.dumps(mufunc, cls=_NPEncoder) + "\n"
                    )
                    mufuncstr = mufuncstr.replace('"', "")
                    writeout.write(mufuncstr)

        return "%s.yace" % name
