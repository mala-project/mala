#!/usr/bin/env python3
import sys
import time
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler


x_keep = 1.0
fp_file = sys.argv[1]
n_components = int(sys.argv[2])
data = np.load(fp_file)
num_fps = np.product(data.shape[:3])
# the final dimension includes spatial (x,y,z) coordinates, which we don't need
num_factors = data.shape[3] - 3
data.shape = (num_fps, num_factors+3)
data = data[:,3:]
keep_index = int(x_keep*num_fps)
data = data[:keep_index,:]
num_fps = data.shape[0]
# Scale the data
scaler = StandardScaler()
data = scaler.fit_transform(data)
# pad the data
num_factors = data.shape[1] # shape may have changed

start = time.perf_counter()
ipca = IncrementalPCA(n_components=n_components, batch_size = None)
ipca.fit(data)
data_transformed = ipca.transform(data) 
end = time.perf_counter()
print("Time: {}".format(end-start))
# Load uncompressed centroids from clustering
#centroids = np.load("centroids.npy")
# Snip off the last column, which was padding in the example I was running
#centroids = centroids[:,:-1]
# Find the scores for the centroids
#centroids_transformed = ipca.transform(centroids)
np.save("transformed_fingerprints.npy",data_transformed)
np.save("fp_components.npy",ipca.components_)
#np.save("transformed_centroids.npy", centroids_transformed)
for v in ipca.explained_variance_ratio_:
    print(v)
