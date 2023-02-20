import numpy as np
import minterpy as mp


####
# Grid
####

def build_cartesian_grid(min_bound,max_bound,shape):
    """
    Generate a cartesian equidistant grid in `[a1,b1]x[a2,b2]x[a3,b3]`,  where `a1,a2,a3 = min_bounds` and `b1,b2,b3 = max_bounds`. 
    The number of grid points per dimension is given by `nx,ny,nz = shape`.
    
    """
    lengths = max_bound - min_bound
    step_size = lengths/(shape+1)
    
    shifted_min = min_bound + step_size
    shifted_max = max_bound - step_size

    # build values for each dimension
    X = np.linspace(shifted_min[0],shifted_max[0],shape[0],endpoint=True)
    Y = np.linspace(shifted_min[1],shifted_max[1],shape[1],endpoint=True)
    Z = np.linspace(shifted_min[2],shifted_max[2],shape[2],endpoint=True)

    # build a batch of all possible combinations, i.e. the cartesian product
    return mp.utils.cartesian_product(X,Y,Z)


def build_local_cells(grid,local_cell_sizes,sim_cell_bounds=None):
    """
    Generate lower and upper boundaries for local cells around each point given in `grid`. The size of each cell is `lx,ly,lz = local_cell_sizes`. 
    If the `sim_cell_bounds` are given, the local cells corresponding to the outer grid points are bounded by these simulation cell bounds.
    
    
    """
    if sim_cell_bounds is None:
        sim_cell_bounds = np.array([np.min(grid,axis=0)- local_cell_sizes,np.max(grid,axis=0) + local_cell_sizes])
    half_norms = local_cell_sizes/2.0
    low_bounds = np.maximum(grid-half_norms.reshape(1,-1),sim_cell_bounds[0].reshape(1,-1))
    high_bounds = np.minimum(grid+half_norms.reshape(1,-1),sim_cell_bounds[1].reshape(1,-1))
    return np.array([low_bounds,high_bounds])


####
# Unisolvent nodes
####


def build_unisolvent_nodes(dim,deg,lp):
    """
    Build unisolvent nodes on the hypercube `[-1,1]^dim` for the polynomial degree `deg` and the degree of the lp-norm `lp`.
    
    The nodes are returned in batch format, i.e. `nodes.shape = (N,dim)`, where `N` is the number of unisolvent nodes. 
    
    Warning: There are no input verification, nor guard rails implemented with this function, except the ones in-place from minterpy itself. Use with caution.
    """
    mi = mp.MultiIndexSet.from_degree(spatial_dimension=dim,poly_degree=deg,lp_degree=lp)
    unisolvent_nodes = mp.Grid(mi).unisolvent_nodes
    return unisolvent_nodes


def transform_domain(unisolvent_nodes,min_bounds,max_bounds):
    """
    Transform unisolvent nodes from `[-1,1]^dim` to `[a1,b1]x[a2,b2]x[a3,b3]`, where `a1,a2,a3 = min_bounds` and `b1,b2,b3 = max_bounds`. 
    
    The nodes passed in are assumed to be in `[-1,1]^dim`, which is *not* checked by this function. 
    
    Warning: There are no input verification, nor guard rails implemented with this function. Use with caution.
    """
    min_bounds = np.require(min_bounds)
    max_bounds = np.require(max_bounds)
    local_nodes = min_bounds + (max_bounds-min_bounds)/(2.0)*(unisolvent_nodes + 1)
    return local_nodes

def transform_to_grid(unisolvent_nodes,min_bounds_grid,max_bounds_grid):
    """
    Copy the `unisolvent_nodes` and trandform the copy to each local cell. 
    
    Warning: It is assumes, that `min_bounds_grid.shape = (N,3)` where `N` denotes the number of grid points, and the last axis contains the respective lower bounds for each dimension. 
    The analogous assumption is made for `max_bounds_grid`.
    
    
    """
    nodes_reshaped = unisolvent_nodes[None,:,:]
    local_min_bounds_reshaped = min_bounds_grid[:,None,:]
    local_max_bounds_reshaped = max_bounds_grid[:,None,:]
    local_unisolvent_nodes = local_min_bounds_reshaped + (local_max_bounds_reshaped-local_min_bounds_reshaped)/(2.0)*(nodes_reshaped + 1)
    return local_unisolvent_nodes


####
# plotting
###

def plot_grid(ax, grid, **kwargs):
    """
    Plot `grid` using the axes object `ax`. The `kwargs` passed in are used to update the default plotting config and then are passed through to the plotting funciton.
    
    Warning: The `grid` should be given in batch format, i.e. `grid.shape = (N,3)`, where `N` is the number of grid points.
    This property is not checked within this function. 
    
    Warning: The axes object `ax` should allow 3d projections, e.g. using `ax = fig.add_subplot(projection='3d')`. Again, this property is not checked in within function.
    """
    # some plotting defaults
    plot_config = {"marker":"+","color":'k'}
    plot_config.update(kwargs)
    
    return ax.scatter(grid[:,0],grid[:,1],grid[:,2],**plot_config)

def plot_local_cell(ax,local_cells, **kwargs):
    """
    Plot the boundaries given in `local_cells`. 
    It is assumed, that `local_cells.shape=(2,N,3)`, where the first axis enumerates the lower and upper bounds,
    the second axis ranges of the grid points and the last axis enumerates the spacial dimentsions.
    
    The `kwargs` passed in are used to update the default plotting config and then are passed through to the plotting funciton.
    
    """
    n_grid = local_cells.shape[1]
    lower_bounds,upper_bounds= local_cells
    
    plot_config = {"alpha":0.1, "color":"blue"}
    plot_config.update(kwargs)
    
    
    if np.log10(n_grid)>=4:
        print("Warning: The number of local cells to plot is very large: <{n_grind}>. This might take a while.")
    
    for idx in np.arange(n_grid):
        lower = lower_bounds[idx]
        upper = upper_bounds[idx]
        ax.plot([lower[0],upper[0]], [lower[1],]*2, [lower[2],]*2,**plot_config)
        ax.plot([lower[0],upper[0]], [upper[1],]*2, [upper[2],]*2,**plot_config)
        ax.plot([lower[0],upper[0]], [lower[1],]*2, [upper[2],]*2,**plot_config)
        ax.plot([lower[0],upper[0]], [upper[1],]*2, [lower[2],]*2,**plot_config)

        ax.plot([lower[0],]*2, [lower[1],upper[1]], [lower[2],]*2,**plot_config)
        ax.plot([upper[0],]*2, [lower[1],upper[1]], [upper[2],]*2,**plot_config)
        ax.plot([lower[0],]*2, [lower[1],upper[1]], [upper[2],]*2,**plot_config)
        ax.plot([upper[0],]*2, [lower[1],upper[1]], [lower[2],]*2,**plot_config)

        ax.plot([lower[0],]*2, [lower[1],]*2, [lower[2],upper[2]],**plot_config)
        ax.plot([upper[0],]*2, [upper[1],]*2, [lower[2],upper[2]],**plot_config)
        ax.plot([lower[0],]*2, [upper[1],]*2, [lower[2],upper[2]],**plot_config)
        ax.plot([upper[0],]*2, [lower[1],]*2, [lower[2],upper[2]],**plot_config)



if __name__ == '__main__':
    __plot__ = True
    
    try:
        import matplotlib.pylab as plt
    except ImportError:
        __plot__=False
    

    DIM,DEG,LP = 3,4,2


    norm = 5.0

    cell_min_bounds = np.zeros(3)
    cell_max_bounds = np.ones(3) * norm
    grid_shape = np.array([3,2,2])

    grid = build_cartesian_grid(cell_min_bounds,cell_max_bounds,grid_shape)



    local_cell_sizes = np.array([0.4,0.4,0.4])

    grid_cells = build_local_cells(grid,local_cell_sizes)
    local_unisolvent_nodes = build_unisolvent_nodes(DIM,DEG,LP)

    grid_unisolvent_nodes = transform_to_grid(local_unisolvent_nodes,grid_cells[0],grid_cells[1])



    if __plot__:
        fig = plt.figure(figsize=(13,10))
        ax = fig.add_subplot(projection='3d')
        ax.azim = 70
        ax.elev = 20
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.scatter(grid[:,0],grid[:,1],grid[:,2],marker="+", c="k")
        plot_local_cell(ax,grid_cells)
        
        for i,pt in enumerate(grid):
            usn = grid_unisolvent_nodes[i]
            ax.scatter(usn[:,0],usn[:,1],usn[:,2],marker=".", c="r")
        plt.show()
