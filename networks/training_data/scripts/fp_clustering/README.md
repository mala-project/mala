# Cluster Fingerprints using PQ-Kmeans

## Purpose
There are two reasons to cluster the data. First, we have a lot of data. Per configuration (point in density x temperature space), we may compute fingerprints at 8e6 spatial locations, and each fingerprint may consist of 116 eight-byte doubles. That's 7.424 GB per configuration, and we plan to use many configurations (order tens or hundreds). That's too much data to train on, so we need some way to choose a (much) smaller but still representative training set for the NN model. Second, NN training can be accelerated and improved if each batch is itself made up of a bunch of dissimilar and representative data.

Because a lot of the data may be similar (imagine a configuration that is a crystal with a point defect -- except for nearby the defect, most of the fingerprints will be repeated over and over, obeying the symmetry of the crystal), purely random sampling likely will not result in a training set that exhibits the variety of the whole dataset. Also, a purely random sample will have a similar distribution to the data themselves, resulting in an overfit to very common fingerprints and not training at all on unusual fingerprints.

The strategy to mitigate the problems of random sampling from the whole dataset is to cluster the data, then sample uniformly from the clusters.

## PQk-means

Naive k-means clustering does not scale well to a million+ samples. PQk-means is an alternative approach [recently proposed](https://yusukematsui.me/project/pqkmeans/pqkmeans.html) for clustering large (billion) scale data. It works by first running naive k-means clustering on a portion of the data. These clusters are then encoded/compressed to a short ( < 10) tuple of integers. Each integer code can take on a short, predefined number of values (say, 0-255). The integer encoding is what is used for clustering, which is vastly accelerated relative to clustering in the original space.

## Approach

One challenge of employing PQk-means on our data is, again, that the data are very large. The authors' implementation of PQk-means is not mpi-parallelized, and the encoding step cannot be performed incrementally. Because fitting even a small fraction of our dataset in memory may be impossible, we can't be sure that the encoder will work well. The approach adopted here is two "bootstrap" use of PQk-means by running it in two phases.

In the first phase, each configuration is individually clustered using PQk-means. These clusters are sampled to produce hopefully representative training data from each configuration. Phase one can be readily parallelized across configurations. Then, in the second phase, these training data are combined and used to train a second encoder. This second encoder is used to cluster data from all configurations.

The memory capacity of one node places a limit on the amount of training data that is sampled from each configuration and used in the second phase. Once the phase-2 encoder has been trained, it can be distributed to multiple nodes, and clustering of the configurations can be accomplished in parallel.

## Instructions

1. Edit the YAML input file
2. mpirun cluster_fingerprints.py. For best results, use one MPI task per node, and use a number of nodes equal to (or that evenly divides into) the total number of configurations.

Input options:

```
configuration root: /scratch/jasteph/cluster_fingerprints/fp_data/
configurations:
  - 300K-gcc0.6.npy
  - 300K-gcc1.0.npy
  - 300K-gcc3.0.npy
Scale: y
workspace: /scratch/jasteph/cluster_fingerprints/workspace
output: /scratch/jasteph/cluster_fingerprints/output
phase one:
  clusters: 64
  codebook size: 256
  dimensions: 4
  samples: 160000
  training fraction: 0.25
  Cluster method: pqkmeans
phase two:
  clusters: 64
  codebook size: 256
  dimensions: 4
  Mismatch test score: 0.01
```

- The location of the configurations (fingerprints) is specified using ```configuration root``` and ```configurations```.
- The ```Scale``` keyword determines whether the fingerprints are scaled (using ```sklearn.preprocessing.StandardScaler```) prior to clustering.
- ```workspace``` and ```output``` are folders where intermediate files and final results files are placed.
- ```phase one``` and ```phase two``` keywords control aspects of the clustering process.
    - ```clusters``` number of clusters
    - ```codebook size``` number of discrete levels that the interger encoding can adopt
    - ```dimensions``` lenght of integer tuple used to represent each sample
    - ```samples``` Total number of samples to grab from each configuration
    - ```training fraction``` fraction of data used for to train the phase-1 encoder
    - ```Mismatch test score``` Used to compute a validation metric. After phase-2 clustering, a "mismatch" is computed for every sample. The mismatch is defined as the sum-squared error between           the cluster center that pqk-means assigned to the sample, and its nearest cluster center. The reported metric is the percentage of samples that have a mismatch smaller than the mismatch           test score.

