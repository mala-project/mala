from mala.descriptors.label_sublib.young import *
from mala.descriptors.gen_labels import *

def ysg_construct_GN(l,L):
    nodeinds, remainderinds = tree(l)
    if remainderinds != None:
        nodes = group_vec_by_orbits(l,tuple([len(nodei) for nodei in nodeinds]) + tuple([1])  )
    else:
        nodes= group_vec_by_orbits(l,tuple([len(nodei) for nodei in nodeinds]) )
    
    if remainderinds == None:
        ysg_pair_per_nodeinds = { node: [] for node in nodeinds}
    else:
        ysg_pair_per_nodeinds = { node: [] for node in nodeinds + tuple([1])}
    ysg_pair_per_node = { node: [] for node in nodes}
    for inode, node in enumerate(nodes):
        ysgi = Young_Subgroup(len(node))
        if len(node) == 2:
            ysgi.subgroup_fill([node[0],node[1]],[(2,),(1,1)],sigma_c_symmetric=False,semistandard=False)
            ysg_pair_per_nodeinds[nodeinds[inode]] = ysgi.fills.copy()
        else:
            ysgi.subgroup_fill([node[0]],[(1,)],sigma_c_symmetric=False,semistandard=False)
            ysg_pair_per_nodeinds[(1,)] = ysgi.fills.copy()
        print (inode,node,ysgi.fills)
        #ysg_pair_per_nodeinds[nodeinds[inode]] = ysgi.fills.copy()
        ysg_pair_per_node[node] = ysgi.fills.copy()
        #print (inode,node,ysgi.fills)

    base_leaf_combs = [p for p in itertools.product(*list(ysg_pair_per_node.values()))]
    print (base_leaf_combs)

    G_N = []
    ysgL = Young_Subgroup(len(L))
    L_inds = [i for i in range(len(l)+1,len(l)+len(L)+1)]
    print (L_inds)
    if len(l) == 4:
        if L[0] == L[1]:
            ysgL.subgroup_fill(L_inds,[(2,),(1,1)],sigma_c_symmetric=False,semistandard=False)
            parent_perms = ysgL.fills.copy()
        elif L[0] != L[1]:
            ysgL.subgroup_fill(L_inds,[(2,)],sigma_c_symmetric=False,semistandard=False)
            parent_perms = ysgL.fills.copy()
    elif len(l) == 5:
        if L[0] == L[1]:
            ysgL.subgroup_fill(L_inds,[(2,1),(1,1,1)],sigma_c_symmetric=False,semistandard=False)
            parent_perms = ysgL.fills.copy()
        elif L[0] != L[1]:
            ysgL.subgroup_fill(L_inds,[(2,1)],sigma_c_symmetric=False,semistandard=False)
            parent_perms = ysgL.fills.copy()

 """   
l=[1,2,3,4]
#l=[1,1,2,2,2]
L = tree_l_inters(l)[0]
ysg_construct_GN(l,L)
"""
