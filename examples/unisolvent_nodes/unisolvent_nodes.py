import numpy as np
import minterpy as mp

def build_unisolvent_nodes(dim,deg,lp):
    """
    Build unisolvent nodes on the hypercube `[-1,1]^dim` for the polynomial degree `deg` and the degree of the lp-norm `lp`.
    
    The nodes are returned in batch format, i.e. `nodes.shape = (N,dim)`, where `N` is the number of unisolvent nodes. 
    
    Warning: There are no input verification, nor guard rails implemented with this function, except the ones in-place from minterpy itself. Use with caution.
    """
    mi = mp.MultiIndexSet.from_degree(spatial_dimension=dim,poly_degree=deg,lp_degree=lp)
    unisolvent_nodes = mp.Grid(mi).unisolvent_nodes
    return unisolvent_nodes


def transform_local_unisolvent_nodes(unisolvent_nodes,min_bounds,max_bounds):
    """
    Transform unisolvent nodes from `[-1,1]^dim` to `[a1,b1]x[a2,b2]x[a3,b3]`, where `a1,a2,a3 = min_bounds` and `b1,b2,b3 = max_bounds`. 
    
    The nodes passed in are assumed to be in `[-1,1]^dim`, which is *not* checked by this function. 
    
    Warning: There are no input verification, nor guard rails implemented with this function. Use with caution.
    """
    min_bounds = np.require(min_bounds)
    max_bounds = np.require(max_bounds)
    local_nodes = min_bounds + (max_bounds-min_bounds)/(2.0)*(unisolvent_nodes + 1)
    return local_nodes

if __name__ == '__main__':
    DIM, DEG, LP = 3,4,2
    unity_nodes = build_unisolvent_nodes(DIM,DEG,LP)
    a = [1,1,1]
    b = [3,3,3]

    transformed_nodes = transform_local_unisolvent_nodes(unity_nodes,a,b)

    print("unity_nodes")
    print(unity_nodes)
    print()

    print("transformed_nodes")
    print(transformed_nodes)
    print()


