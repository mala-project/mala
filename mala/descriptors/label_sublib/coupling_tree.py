import math
from mala.descriptors.gen_labels import * 

parity_parts_per_rank =      {  4: [(4,), (2,2) ],
                5: [(5,), (4,1), (2,2,1) ],
                6: [(6,), (4,2), (2,2,2) ],
                7: [(7,), (6,1), (4,2,1), (2,2,2,1) ],
                8: [(8,), (6,2), (4,4), (4,2,2), (2,2,2,2) ]
}
class Jucy_Tree:
    def __init__(self,l,L,L_R):
        #self.l=l
        self.children = {0:l}
        self.L = L
        self.rank = len(l)
        self.all_by_orbits = None
        self.orbit = None

    def add_child_nodes(self,v):
        current_children = self.children
        current_child_keys = list(current_children.keys())
        new_child_key = len(current_child_keys) 
        current_children[new_child_key] = v
        self.children = current_children

    def group_children(self,part='max_nc2'):
        if type(part) != tuple and part =='max_nc2':
            nc2 = math.floor(self.rank/2)
            if (nc2 * 2) != self.rank:
                remain = (1,)
            else:
                remain = ()
            part = tuple ([2] * nc2) + remain
            trivial_part = tuple([1]*self.rank)
        self.orbit = part
        children_list = list(self.children.values())
        children_groups = {}
        children_trivial = {}
        for child_ind,child in self.children.items():
            grouped = group_vec_by_orbits(child,part)
            grouped_triv = group_vec_by_orbits(child,trivial_part)
            children_groups[child_ind] = grouped
            children_trivial[child_ind] = grouped
        self.child_orbits = children_groups
        self.child_nosym = children_trivial

        child_orbit_list = list(children_groups.values())
        self.all_by_orbits = [v for v in zip(*child_orbit_list)]

        child_trivial_list = list(children_trivial.values())
        self.all_by_trivial = [v for v in zip(*child_trivial_list)]

    def return_id(self):
        assert self.all_by_orbits != None, "you must group children nodes by orbits before getting an id. run Jucy_Tree.group_children() and try again"
        this_id = [ ( Li, )+ children for Li, children in zip(self.L,self.all_by_orbits) ]
        print (this_id)
        return tuple(this_id)

    def set_coupling_parities(self):
        assert self.all_by_orbits != None, "you must group children nodes by orbits before getting an id. run Jucy_Tree.group_children() and try again"

        reducible_children = {}
        children_base_degen = {}
        children_base_parity = {}
        this_map = {True:'g',False:'u'}
        for child_ind,child_orbit_set in self.child_orbits.items():
            child_base_pars = [ child_orbit.count(child_orbit[0]) == len(child_orbit) for child_orbit in child_orbit_set]
            child_base_reducible = child_base_pars.count(child_base_pars[0]) == len(child_base_pars)
            reducible_children[child_ind] = child_base_reducible
            children_base_degen[child_ind] = child_base_pars
            children_base_parity[child_ind] = [this_map[cp] for cp in child_base_pars]
        self.base_degen_per_child = children_base_degen
        self.base_parity_per_child = children_base_parity

