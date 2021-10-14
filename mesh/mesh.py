from .mesh_utils import *
import numpy as np
import os
import networkx as nx

class icosphere(object):

    def __init__(self, level=0, unpool=False):
        """
        Self-vertices included in pooling indices, but not graph construction
        """
        self.level = level
        self.vertices, self.faces = self.icosahedron()
        self.nf, self.nv = self.faces.shape[0], self.vertices.shape[0]
        self.ne = 30 * (4 ** self.level)

        self.knn = None
        self.v0, self.f0 = self.vertices.copy(), self.faces.copy()
        
        g0 = self.construct_graph()
        nv0 = len(self.v0)
        self.graphs_by_level = [g0]
        # V_l -> V_l+1 neighbor indices
        self.base_pool_inds = []
        self.rest_pool_inds = []
        self.unpool_inds = [] # make optional
        for l in range(1, self.level+1):
            self.subdivide()
            self.normalize()
            g = self.construct_graph()
            self.graphs_by_level.append(g)
            base_pool_ind, rest_pool_ind = self.pool_indices(g, l-1)
            self.base_pool_inds.append(base_pool_ind)
            self.rest_pool_inds.append(rest_pool_ind)
            # unpooling
            if unpool:
                pool_ind = list(base_pool_ind)
                if l > 1:
                    pool_ind.extend(rest_pool_ind) 
                unpool_ind = self.unpool_indices(pool_ind, l-1)
                self.unpool_inds.append(unpool_ind)

        for l, g in enumerate(self.graphs_by_level):
            print("l={}, |V|={}, |E|={}".format(l, g.number_of_nodes(), g.number_of_edges()))

        self.graph = self.graphs_by_level[-1]
        self.nf, self.nv = self.faces.shape[0], self.vertices.shape[0]
        self.ne = 30 * (4 ** self.level)
        self.nv = self.ne - self.nf + 2

    def pool_indices(self, g, l_prev):
        deg_base, deg_rest = 5, 6
        nv0 = self.v0.shape[0]
        nv = self.num_vertices_by_level(l_prev)
        v_g = np.arange(g.number_of_nodes()) # graph nodes, ordered
        nv_rest = nv - nv0
        base_ind = np.full((nv0, deg_base), -1)
        base_neighb_ids = np.array([g[i] for i in range(nv0)])
        base_ind[:nv0] = base_neighb_ids[:]

        # add self vertices
        base_v = np.arange(nv0)
        base_pool_ind = base_ind
        base_pool_ind = np.concatenate((base_v[:,None], base_ind), axis=1)

        if nv_rest: # empty for L1 -> L0 pooling
            rest_ind = np.full((nv_rest, deg_rest), -1)
            rest_neighb_ids = np.array([g[i] for i in range(nv0, nv)])
            rest_ind[:nv_rest] = rest_neighb_ids[:]
            rest_v = np.arange(nv0,nv)
            rest_pool_ind = rest_ind
            rest_pool_ind = np.concatenate((rest_v[:,None], rest_ind), axis=1)
        else:
            rest_pool_ind = None
        
        return base_pool_ind, rest_pool_ind

    def unpool_indices(self, pool_ind, l_prev):
        nv_prev = self.num_vertices_by_level(l_prev)
        nv_next = self.num_vertices_by_level(l_prev+1)
        unpool_ind = [[] for _ in range(nv_next-nv_prev)]
        for u, neighbor_ids in enumerate(pool_ind):
            for v in neighbor_ids:
                v_idx = v - nv_prev
                if v_idx >= 0: #ignore negative indices from self-neighbors
                    unpool_ind[v_idx].append(u)
        unpool_ind = np.array(unpool_ind, dtype=int)
        return unpool_ind
        

    def subdivide(self):
        """
        Subdivide a mesh into smaller triangles.
        """
        faces = self.faces
        vertices = self.vertices
        face_index = np.arange(len(faces))

        # the (c,3) int set of vertex indices
        faces = faces[face_index]
        # the (c, 3, 3) float set of points in the triangles
        triangles = vertices[faces]
        # the 3 midpoints of each triangle edge vstacked to a (3*c, 3) float
        src_idx = np.vstack([faces[:, g] for g in [[0, 1], [1, 2], [2, 0]]])
        mid = np.vstack([triangles[:, g, :].mean(axis=1) for g in [[0, 1],
                                                                   [1, 2],
                                                                   [2, 0]]])
        mid_idx = (np.arange(len(face_index) * 3)).reshape((3, -1)).T

        # for adjacent faces we are going to be generating the same midpoint
        # twice, so we handle it here by finding the unique vertices
        unique, inverse = unique_rows(mid)
        mid = mid[unique]
        src_idx = src_idx[unique]
        mid_idx = inverse[mid_idx] + len(vertices)

        # the new faces, with correct winding
        f = np.column_stack([faces[:, 0], mid_idx[:, 0], mid_idx[:, 2],
                             mid_idx[:, 0], faces[:, 1], mid_idx[:, 1],
                             mid_idx[:, 2], mid_idx[:, 1], faces[:, 2],
                             mid_idx[:, 0], mid_idx[:, 1], mid_idx[:, 2], ]).reshape((-1, 3))
        # add the 3 new faces per old face
        new_faces = np.vstack((faces, f[len(face_index):]))
        # replace the old face with a smaller face
        new_faces[face_index] = f[:len(face_index)]

        new_vertices = np.vstack((vertices, mid))

        self.vertices = new_vertices
        self.faces = new_faces
    
    def normalize(self, radius=1):
        '''
        Reproject to spherical surface
        '''
        vectors = self.vertices
        scalar = (vectors ** 2).sum(axis=1)**.5
        unit = vectors / scalar.reshape((-1, 1))
        offset = radius - scalar
        self.vertices += unit * offset.reshape((-1, 1))
        
    def icosahedron(self):
        """
        Create an icosahedron, a 20 faced polyhedron.
        """
        t = (1.0 + 5.0**.5) / 2.0
        vertices = [-1, t, 0, 1, t, 0, -1, -t, 0, 1, -t, 0, 0, -1, t, 0, 1, t,
                    0, -1, -t, 0, 1, -t, t, 0, -1, t, 0, 1, -t, 0, -1, -t, 0, 1]
        faces = [0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
                 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
                 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
                 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1]
        # make every vertex have radius 1.0
        vertices = np.reshape(vertices, (-1, 3)) / 1.9021130325903071
        faces = np.reshape(faces, (-1, 3))

        return vertices, faces

    def construct_graph(self):
        """
        Construct undirected graph over vertices
        """
        edges = []
        for v1, v2, v3 in self.faces:
            edges.append((v2, v3))
            edges.append((v1, v3))
            edges.append((v1, v2))
        
        g = nx.Graph(edges)
        for v in g.nodes():
            assert (g.degree[v] == 5 or g.degree[v] == 6), "Invalid neighbors {}".format(g.degree[v])

        return g


    @staticmethod
    def num_vertices_by_level(l):
        return 10*(4**l)+2
