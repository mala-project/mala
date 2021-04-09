#!/usr/bin/env python3
import os
import sys
import glob
import math
from collections import defaultdict
from itertools import cycle
import yaml
import numpy as np
import scipy.stats
from sklearn.preprocessing import StandardScaler
import sklearn.cluster
from mpi4py import MPI
import pqkmeans
import sympy


mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

def log(message, ranks='root'):
    """Print a message

    ranks: 'all' or 'root'
    """
    global mpi_rank
    if ranks == 'all':
        print("Rank %d: " % mpi_rank, message,flush = True)
    elif ranks == 'root':
        if mpi_rank == 0:
            print(message,flush=True)
               

def reshape_fps(X):
    """Reshape 4D fingerprint data to 2D
 
    If X is already 2D, do nothing.
  
    Returns: reshaped X
    """
    if len(X.shape) == 4:
        num_factors = X.shape[3]
        num_fps = np.prod(X.shape[:3])
        X.shape = (num_fps,num_factors)
    else:
        num_factors = X.shape[1]
        num_fps = X.shape[0]
    return X

def load_and_scale_fps(config_file, scaler):
    """Load and scale the 4D fingerprints array at config_file

    If scaler is None, do not scale.
    """
    X = np.load(config_file)
    x_shape = X.shape
    X = reshape_fps(X)
    if scaler:
        X = scaler.transform(X)
    X.shape = x_shape
    return X

def pq_encode(X_in, codebook_size, dimensions,training_fraction):
    """Construct and return a PQk-means encoder

    X_in: A 4D numpy array that contains the fingerprints
    codebook_size: number of discrete levels in the codebook
    dimension: Length of compressed sample. Must evenly divide number of descriptors
    training_fraction: Fraction of data in X_in to use as training data to build the encoder

    Returns: an encoder
    """
    X = np.array(X_in, copy=True)
    x_shape = X.shape
    X = reshape_fps(X)
    num_fps, num_factors = X.shape
    train_samples = int(num_fps * training_fraction)
    np.random.shuffle(X)
    encoder = pqkmeans.encoder.PQEncoder(num_subdim=dimensions, Ks=codebook_size)
    encoder.fit(X[:train_samples,:])
    X.shape = x_shape
    return encoder

def pq_cluster(X, encoder, num_clusters):
    """Cluster the data in X using the encoder

    X: 4D numpy array of data to be clustered
    encoder: PQk-means encoder to use for clustering
    num_clusters: number of clusters

    returns: a 3D numpy array that contains the indexes of the clusters that the data in X belong to
    """
    x_shape = X.shape
    X = reshape_fps(X)
    num_fps,num_factors = X.shape
    dimensions = encoder.codewords.shape[0]
    pqcodes = np.empty([num_fps, dimensions], dtype=encoder.code_dtype)
    # Encode all fingerprints
    pqcodes = encoder.transform(X)
    pqk = pqkmeans.clustering.PQKMeans(encoder=encoder, k=num_clusters)
    clusters = np.array(pqk.fit_predict(pqcodes))
    cluster_centers = np.array(pqk.cluster_centers_, dtype=encoder.code_dtype)
    X.shape = x_shape
    clusters.shape = x_shape[0:3]
    return clusters, cluster_centers

def sklearn_kmeans_cluster(X, num_clusters):
    """Use sklearn.cluster.KMeans to do phase 1 clustering"""
    x_shape = X.shape
    X = reshape_fps(X)
    num_fps, num_factors = X.shape
    km = sklearn.cluster.KMeans(n_clusters=num_clusters)
    clusters = km.fit_predict(X)
    X.shape = x_shape
    return clusters

def sklearn_minibatchkmeans_cluster(X_in, num_clusters):
    """Use sklearn.cluster.KMeans to do phase 1 clustering"""
    X = np.array(X_in,copy=True)
    X = reshape_fps(X)
    num_fps, num_factors = X.shape
    km = sklearn.cluster.MiniBatchKMeans(n_clusters=num_clusters, batch_size=1000, max_no_improvement=None)
    clusters = km.fit_predict(X)
    return clusters


def write_samples(clusters, X, num_samples, fp_index, path):
    """Write out samples drawn iid from the clusters

    clusters: 3D array of cluster indexes
    X: 4D array of clustered data
    num_samples: Total number of samples to write out
    fp_index: fingerprint file index
    path: workspace directory path
    """
    binned = defaultdict(list)
    x_shape = X.shape
    X = reshape_fps(X)
    num_fps, num_factors = X.shape
    clusters.shape = (num_fps,)

    for i, cluster in enumerate(clusters):
        binned[cluster].append(i)
    for c in binned:
        np.random.shuffle(binned[c]) # clustered may be ordered somewhow. 
    num_clusters = len(binned)
    # Grab samples from the clusters in round-robin fashion until we've 
    # collected num_train
    tctr = 0 # number of samples we've collected so far
    X_samples = np.empty(shape=(num_samples,num_factors)) # store collected 
                                                          # samples here
    for c in cycle(binned.keys()):
        if tctr == num_samples:
            break
        try:
            i = binned[c].pop() # cluster c could run empty, raising IndexError
            X_samples[tctr,:] = X[i,:]
            tctr += 1
        except IndexError:
            pass
    samples_file = os.path.join(path, "training_samples_%03d.npy" % fp_index)
    np.save(samples_file,X_samples)

def load_samples(path):
    """Load the samples that are to be used to train a PQk-means encoder"""

    config_sample_files = glob.glob(path + os.sep + "training_samples_*.npy")
    all_samples = []
    for c in config_sample_files:
        all_samples.append(np.load(c))
    return np.concatenate(all_samples,axis=0)

def write_clusters(clusters, cluster_centers, fp_index, path):
    """Write the clusters to disk"""
    clusters_file = os.path.join(path, "clusters_%03d.npy" % fp_index)
    np.save(clusters_file, clusters)
    centers_file = os.path.join(path, "centers_%03d.npy" % fp_index)
    np.save(centers_file, cluster_centers) 

def write_codewords(encoder, scaler, path):
    """Write the codewords for the encoder"""

    cw = np.array(encoder.codewords,copy=True)
    if scaler:
        num_dim, num_lvls, dim_size = cw.shape
        cw2 = np.empty((num_lvls, num_dim*dim_size))
        for i in range(num_dim): 
            cw2[:,i*dim_size:(i+1)*dim_size] = cw[i,:,:]
        cw2 = scaler.inverse_transform(cw2)
        for i in range(num_dim):
            cw[i,:,:] = cw2[:,i*dim_size:(i+1)*dim_size]
    cw_file = os.path.join(path,"p2_codewords.npy")
    np.save(cw_file,cw)

def compute_mismatch(encoder, X, labels, centers, score):
    """Return the fraction of mislabeled points in X

    For each point in X, determine the nerest center in centers. Then, compute
    the mismatch between the labeled center and this actual nearest one. The
    mistmatch is the sum-squared error. Return the fraction of mismatches
    that are less than the score."""
    x_shape = X.shape
    X = reshape_fps(X)
    num_fps,num_factors = X.shape
    labels.shape = (num_fps,)
    # inv_centers is the coordinates of the cluster centers in the original
    # space 
    inv_centers = encoder.inverse_transform(centers)
    mismatches = np.zeros((num_fps,))
    
    total_dist = 0

    for i in range(num_fps):
        distances = np.sum((X[i] - inv_centers)**2.0,axis=1)
        argmin = np.argmin(distances)
        mismatches[i] = np.sum((inv_centers[labels[i]] - 
                                inv_centers[argmin])**2.0)

        # Total sum of distances from sample to center
        total_dist += np.sum((X[i] - inv_centers[labels[i]]) ** 2.0)

    print("Mismatch sum: %4.4f" % np.sum(mismatches))
    print("Distance sum: %4.4f" % total_dist)
#    X.shape = x_shape
#    labels.shape = x_shape[:3]
    return scipy.stats.percentileofscore(mismatches,score)

##############

def cluster_snapshot(path, num_samples, shape, subset, scaler, clusters, train_ratio):

    fp = np.load(path)
    fp = np.reshape(fp, shape)
    fp = fp[:, subset]
    fp = scaler.do_scaling_sample(fp)

    power2 = int(np.ceil(np.log2(fp.shape[1])))
    nsbdm = int(2 ** np.floor(power2 / 2.0))

    # Pad vector to be a power of 2
    fp_padded = np.zeros([fp.shape[0], 2 ** power2])

    fp_padded[:fp.shape[0], :fp.shape[1]] = fp

    np.random.shuffle(fp_padded)

    # find divisor 
    #factors = sympy.ntheory.factorint(fp.shape[1])
    #f_keys = list(factors.keys())

    #nsbdm = int(f_keys[0] * factors[int(f_keys[0])])
    ks = 256

    print("Begin Encoder training with num_subdim %d" % (nsbdm))
    encoder = pqkmeans.encoder.PQEncoder(num_subdim=nsbdm, Ks=ks) 
    
    fit_samples = int(num_samples * train_ratio)

    fit_mask = np.zeros([num_samples], dtype=bool)
    fit_mask[:fit_samples] = True
    fit_mask = np.random.permutation(fit_mask)

    encoder.fit(fp_padded[fit_mask])

    print("Begin fp transform")
    fp_pqcode = encoder.transform(fp_padded)

    del fp
    del fp_padded

    print("Begin Kmeans set")
    kmeans = pqkmeans.clustering.PQKMeans(encoder=encoder, k=clusters)
    
    print("Begin Kmeans fit predict")
    return kmeans.fit_predict(fp_pqcode)


##############

def main(input_file):
   
    global mpi_comm
    global mpi_rank
    global mpi_size

    with open(input_file,"r") as f:
        options = yaml.safe_load(f)
    configs = options["configurations"]
    config_root = options["configuration root"]
    num_configs = len(configs)
    configs_per_rank = math.ceil(num_configs / mpi_size)
    # TODO: this is fragile. Probably breaks when there are fewer ranks than 
    # configs?
    scale = options["Scale"]

    # Fit a scaler on all the data if requested
    if scale:
        log("Begin scaling..")
        if mpi_rank == 0:
            scaler = StandardScaler(copy=False)
            for config in configs:
                config_file = os.path.join(config_root,config)
                X = np.load(config_file)
                X = reshape_fps(X)
                scaler.partial_fit(X)
                del X
            mpi_comm.bcast(scaler,root=0)
        else:
            scaler = mpi_comm.bcast(None,root=0)
        log("Scaling complete.")
    else:
        scaler = None
    # Phase 1 - Cluster each configuration separately to generate training
    # data.
    p1_opts = options["phase one"]
    log("Beginning Phase 1.")
    for config in configs[mpi_rank::mpi_size]:
        config_file = os.path.join(config_root,config)
        X = load_and_scale_fps(config_file, scaler)
        if p1_opts["Cluster method"] == "pqkmeans":
            log("Encoding config %s" % config,"all")
            encoder = pq_encode(X, p1_opts["codebook size"],
                                 p1_opts["dimensions"],
                                 p1_opts["training fraction"])
            clusters, _ = pq_cluster(X, encoder, p1_opts["clusters"])
            log("Clustering config %s" % config,"all")
            del encoder
        elif p1_opts["Cluster method"] == "sklearn.kmeans":
            clusters = sklearn_kmeans_cluster(X, p1_opts["clusters"])
        elif p1_opts["Cluster method"] == "sklearn.minibatchkmeans":
            clusters = sklearn_minibatchkmeans_cluster(X, p1_opts["clusters"])
        write_samples(clusters, X, p1_opts["samples"],
                      configs.index(config), options["workspace"])
        del X, clusters
        #
        # clusters = pqkmeans.clustering.PQKMeans(
        #    encoder=encoder, k=p1_opts["clusters"]).fit_predict(pqcodes)
    mpi_comm.Barrier()
    log("Phase 1 Complete. Beginning Phase 2.")
    
    # Phase 2 - Train an encoder using data from Phase 1 and cluster all data
    p2_opts = options["phase two"]
    if mpi_rank == 0:
        samples = load_samples(options["workspace"])
        encoder = pq_encode(samples, p2_opts["codebook size"],
                             p2_opts["dimensions"],1.0)
        log("Phase 2 encoder trained.")
        write_codewords(encoder, options["output"])
        del samples
        mpi_comm.bcast(encoder, root=0)
    else:
        encoder = mpi_comm.bcast(None, root=0)
    
    percentiles = np.zeros((configs_per_rank,))
    for i, config in enumerate(configs[mpi_rank::mpi_size]):
        log("Clustering config %s" % config, "all")
        config_file = os.path.join(config_root,config)
        X = load_and_scale_fps(config_file, scaler)
        clusters, cluster_centers = pq_cluster(X, encoder, p2_opts["clusters"])
        write_clusters(clusters, cluster_centers, configs.index(config),
                        options["output"])
        percentiles[i] = compute_mismatch(encoder, X, clusters,
                                           cluster_centers, 
                                           p2_opts["Mismatch test score"])
        log("Percentile for config %s: %g" % (config, percentiles[i]),"all")
        del X, clusters, encoder 
    if mpi_rank == 0:
        totals = np.zeros_like(percentiles)
        mpi_comm.Reduce([percentiles, MPI.DOUBLE], [totals, MPI.DOUBLE],
                     op=MPI.SUM, root=0)
        p_sum = np.sum(totals)
        log("Average percentile: %g" % (p_sum/num_configs,))
    else:
        totals = None
        mpi_comm.Reduce([percentiles,MPI.DOUBLE], totals,  op=MPI.SUM,root=0)
    
if __name__ == '__main__':
    main(sys.argv[1])
