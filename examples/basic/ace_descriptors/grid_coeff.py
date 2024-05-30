from yamlpace_tools.potential import *
from pa_gen import *
from yamlpace_tools.grid_defaults import *
from tree_method import *
from clebsch_couple import *
from wigner_couple import *


#---------------------------------------------------------------------
# Logical flags that define density to be expanded with ACE
#---------------------------------------------------------------------

# logical flag to filter off ACE basis functions that include
#   explicit interactions with other grid points
grid_filter = True
# logical flag to enforce that the LAMMPS type assigned to grid 
#   points is the same as type 1 (note this is only compatible
#   with grid_filter = False)
types_like_snap = False

# logical flag to add 1 padding function for each element type != grid
#   type. This is required if "grid_filter = True"
padfunc = True
#---------------------------------------------------------------------
# define descriptor set and hyperparameters
#---------------------------------------------------------------------


# N-bond functions to enumerate N is rank
ranks = [1,2,3]
#chemical basis NOTE that the last element type should be assigned 'G' for the grid point
elements = ["Al", 'G']
reference_ens = [0.0,0.0]
mumax = len(elements)
# radial basis
nmax = [6,2,2] # max radial index per rank
nradbase=max(nmax)
#angular basis
lmax = [0,2,2] # max angular index per rank
# lmax = 0 for first rank only
lmin = [0,0,0] # minimum angular index per rank (keep 0 as default for complete basis)
L_R=0
M_R=0

# radial function hyperparameters
# NOTE these are defined per bond type
# bond types are obtained from [p for p in itertools.product(elements,elements)]
bonds = [p for p in itertools.product(elements,elements)]
#rc_range,rc_default,lmb_default,rcin_default = get_default_settings(elems,nshell=1.8,return_range=True,apply_shift=False)
rc_range,rc_default,lmb_default,rcin_default = get_default_settings(elements,nshell=2,return_range=True,apply_shift=False)
#inner cutoff if needed
rcutfac= [float(k) for k in rc_default.split()[2:]]
lmbda=[float(k) for k in lmb_default.split()[2:]]
assert len(bonds) == len(rcutfac) and len(bonds) == len(lmbda),"you must have rcutfac and lmbda defined for each bond type"
print('global max cutoff (angstrom)',max(rcutfac))
rcinner = [0.0] * len(bonds)
drcinner = [0.0]* len(bonds)

#---------------------------------------------------------------------
# Generate generalized Clebsch-Gordan coefficients or Wigner symbols
#     used for combining products of spherical harmonics
#---------------------------------------------------------------------
#coupling type for angular spherical harmonics (either is fine)
coupling_type ='cg'
#wigner couplings (instead of Clebsch-Gordan) above

coeffs = None
ldict = {ranki:li for ranki,li in zip(ranks,lmax)}
rankstrlst = ['%s']*len(ranks)
rankstr = ''.join(rankstrlst) % tuple(ranks)
lstrlst = ['%s']*len(ranks)
lstr = ''.join(lstrlst) % tuple(lmax)

#load or generate generalized coupling coefficients
if coupling_type == 'cg':
    try:
        with open('cg_LR_%d_r%s_lmax%s.pickle' %(L_R,rankstr,lstr),'rb') as handle:
            ccs = pickle.load(handle)
    except FileNotFoundError:
        ccs = get_cg_coupling(ldict,L_R=L_R)
        #store them for later so they don't need to be recalculated
        store_generalized(ccs, coupling_type='cg',L_R=L_R)

elif coupling_type == 'wig':
    try:
        with open('wig_LR_%d_r%s_lmax%s.pickle' %(L_R,rankstr,lstr),'rb') as handle:
            ccs = pickle.load(handle)
    except FileNotFoundError:
        ccs = get_wig_coupling(ldict,L_R=L_R)
        #store them for later so they don't need to be recalculated
        store_generalized(ccs, coupling_type='wig',L_R=L_R)


#---------------------------------------------------------------------
# generate descriptor labels and sort them for lammps input
#---------------------------------------------------------------------

def srt_by_attyp(nulst,remove_type=2):
    mu0s = []
    for nu in nulst:
        mu0 = nu.split('_')[0]
        if mu0 not in mu0s:
            mu0s.append(mu0)
    mu0s = sorted(mu0s)
    byattyp = {mu0:[] for mu0 in mu0s}
    byattypfiltered = {mu0:[] for mu0 in mu0s}
    mumax = remove_type - 1
    for nu in nulst:
        mu0 = nu.split('_')[0]
        byattyp[mu0].append(nu)
        mu0ii,muii,nii,lii = get_mu_n_l(nu)
        if mumax not in muii:
            byattypfiltered[mu0].append(nu)
    return byattyp,byattypfiltered

ranked_chem_nus = []
for ind,rank in enumerate(ranks):
    rank = int(rank)
    PA_lammps, not_compat = pa_labels_raw(rank,int(nmax[ind]),int(lmax[ind]), int(mumax),lmin = int(lmin[ind]) )
    ranked_chem_nus.append(PA_lammps)

nus_unsort = [item for sublist in ranked_chem_nus for item in sublist]
nus = nus_unsort.copy()
mu0s = []
mus =[]
ns = []
ls = []
for nu in nus_unsort:
    mu0ii,muii,nii,lii = get_mu_n_l(nu)
    mu0s.append(mu0ii)
    mus.append(tuple(muii))
    ns.append(tuple(nii))
    ls.append(tuple(lii))
nus.sort(key = lambda x : mus[nus_unsort.index(x)],reverse = False)
nus.sort(key = lambda x : ns[nus_unsort.index(x)],reverse = False)
nus.sort(key = lambda x : ls[nus_unsort.index(x)],reverse = False)
nus.sort(key = lambda x : mu0s[nus_unsort.index(x)],reverse = False)
nus.sort(key = lambda x : len(x),reverse = False)
nus.sort(key = lambda x : mu0s[nus_unsort.index(x)],reverse = False)
musins = range(len(elements)-1)
all_funcs = {}
if types_like_snap:
    byattyp,byattypfiltered = srt_by_attyp(nus,1)
    if grid_filter:
        assert padfunc,"must pad with at least 1 other basis function for other element types to work in LAMMPS - set padfunc=True"
        limit_nus = byattypfiltered['%d'%0]
        if padfunc:
            for muii in musins:
                limit_nus.append(byattypfiltered['%d'%muii][0])
    elif not grid_filter:
        limit_nus = byattyp['%d'%0]

elif not types_like_snap:
    byattyp,byattypfiltered = srt_by_attyp(nus,len(elements))
    if grid_filter:
        limit_nus = byattypfiltered['%d'%(len(elements)-1)]
        assert padfunc,"must pad with at least 1 other basis function for other element types to work in LAMMPS - set padfunc=True"
        if padfunc:
            for muii in musins:
                limit_nus.append(byattypfiltered['%d'%muii][0])
    elif not grid_filter:
        limit_nus = byattyp['%d'%(len(elements)-1)]
        if padfunc:
            for muii in musins:
                limit_nus.append(byattyp['%d'%muii][0])
print('all basis functions',len(nus),'grid subset',len(limit_nus))


#permutation symmetry adapted ACE labels
Apot = AcePot(elements,reference_ens,ranks,nmax,lmax,nradbase,rcutfac,lmbda,rcinner,drcinner,lmin=lmin, **{'input_nus':limit_nus,'ccs':ccs[M_R]})
Apot.write_pot('coupling_coefficients_fullbasis')

Apot.set_funcs(nulst=limit_nus,muflg=True,print_0s=True)
Apot.write_pot('coupling_coefficients')
#with open('ctildeslimit.json','w') as writejson:
#    json.dump(all_funcs,writejson, sort_keys=False, indent=2)
