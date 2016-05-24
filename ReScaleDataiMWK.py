#!/usr/env Python
import clustering
import numpy as np
import mdtraj as md
import sklearn.metrics
import sklearn.cluster
import matplotlib
matplotlib.use('Agg') #For use on DEAC cluster
import matplotlib.pyplot as plt
plt.style.use('bmh')
import argparse
import scipy
import copy

#Initialize parser. The default help has poor labeling. See http://bugs.python.org/issue9694
parser = argparse.ArgumentParser(description = 'Compare Minkowski weights', add_help=False) 

#List all possible user input
inputs=parser.add_argument_group('Input arguments')
inputs.add_argument('-h', '--help', action='help')
inputs.add_argument('-top', action='store', dest='structure',help='Structure file corresponding to trajectory',type=str,required=True)
inputs.add_argument('-traj', action='store', dest='trajectory',help='Trajectory',type=str,required=True)
inputs.add_argument('-sel', action='store', dest='sel', help='Atom selection',type=str,default='not element H')
inputs.add_argument('-o', action='store', dest='out_name',help='Output file',type=str,required=True)

#Parse into useful form
UserInput=parser.parse_args()


topology = UserInput.structure
trajectory = UserInput.trajectory
t = md.load(trajectory,top=topology)
sel = t.topology.select(UserInput.sel)
t = t.atom_slice(sel)

# Format trajectory 
temp = t.xyz
frames = t.xyz.shape[0]
atoms = t.xyz.shape[1]
original_data = temp.reshape((frames,atoms*3))
original_data = original_data.astype('float64')
temp = []

t = []
#Figure out what P is 
np.seterr(all='raise')
cl = clustering.Clustering()
if frames > 10000:
    sample_size = 10000
else:
    sample_size = None

original_data = cl.my_math.standardize(original_data) #Not clear if I should do this

# Trying to find the optimal p
#data = copy.copy(original_data)
#data = cl.my_math.standardize(data) #Not clear if I should do this
#p_to_try = np.arange(1.1,5.1,0.1) #Amorim's suggestion
#silhouette_scores = np.zeros(p_to_try.size)
#for q in range(0, p_to_try.size):
#    print('Testing Minkowski Weight ' + str(p_to_try[q]) + ' with max of 5.0')
#    [u, centroids, weights, ite, dist_tmp] = cl.imwk_means(original_data, p_to_try[q], gradient=np.array([0.1]), cutoff=0.1)
    # Calculate silhouette score
#    silhouette_scores[q] = sklearn.metrics.silhouette_score(original_data,u)

#data=[]

# Store optimal p
#optimal_p = p_to_try[np.argmax(silhouette_scores)]

# Searching for p has proved computationally intractable - RLM
optimal_p=2 #From Amorim's experiments
# Ready to do iMWK-means with explicit rescaling

# Set an upper bound on k
[labels, centroids, weights, ite, dist_tmp] = cl.imwk_means(original_data, p=optimal_p)
maxk = max(labels) + 1 #Cluster labels start at 0
silhouette_averages = np.zeros(maxk-1)

print("Rescale iMWK trials")
k_to_try = np.arange(2,maxk+1)
for k in k_to_try:
    print('Testing k=' + str(k) + ' of ' +  str(maxk))
    cl=[]
    labels = []
    weights = []
    centroids = []
    cl=clustering.Clustering()
    data = copy.copy(original_data)
    [labels, centroids, weights, ite, dist_tmp] = cl.imwk_means(data, p=optimal_p,k=k)
    # For matching matlab implementation
    # W = weights
    # Z = centroids
    # U = labels

    # Rescale the data
    for k1 in np.arange(0,max(labels)+1):
        data[labels==k1] = np.multiply(data[labels==k1],np.tile(weights[k1], (np.sum(labels==k1),1)))
        centroids[k1] = np.multiply(centroids[k1], weights[k1])

    # Apply Euclidean KMeans
    kmeans_clusterer = sklearn.cluster.KMeans(n_clusters=k, n_init=100, n_jobs=-1)
    kmeans_clusters = kmeans_clusterer.fit(data)
    labels = kmeans_clusters.labels_
    centroids = kmeans_clusters.cluster_centers_

    # Do I want to also do KMeans city block? Amorim's code does it (L1), but that's not discussed in his paper...
    # Also, I'm calculating my CVIs on the rescaled data ... that might be cheating

    # Silhouette scores are roughly matching matlab (stochastic, so will never be exact)
    silhouette_averages[k - 2] = sklearn.metrics.silhouette_score(data, labels, sample_size=sample_size)

optimal_k = k_to_try[np.argmax(silhouette_averages)]
# Do optimal clustering
cl=[]
labels = []
weights = []
centroids = []
cl=clustering.Clustering()
data = copy.copy(original_data)
[labels, centroids, weights, ite, dist_tmp] = cl.imwk_means(data, p=optimal_p, k=optimal_k)
# Rescale the data
for k1 in np.arange(0,max(labels)+1):
    data[labels==k1] = np.multiply(data[labels==k1],np.tile(weights[k1], (np.sum(labels==k1),1)))
    centroids[k1] = np.multiply(centroids[k1], weights[k1])

# Apply Euclidean KMeans
kmeans_clusterer = sklearn.cluster.KMeans(n_clusters=optimal_k, n_init=100, n_jobs=-1)
kmeans_clusters = kmeans_clusterer.fit(data)
labels = kmeans_clusters.labels_
centroids = kmeans_clusters.cluster_centers_
silhouette_score = sklearn.metrics.silhouette_score(data, labels, sample_size=sample_size)

np.savetxt(UserInput.out_name + '/RescalediMWK_labels.txt', labels, fmt='%i')
with open (UserInput.out_name + '/silhouette_score.txt', 'w') as f:
    f.write("silhouette score is {0} \n with p of {1}\n".format(silhouette_score,optimal_p))

#Figures
plt.figure()
plt.scatter(np.arange(frames), labels, marker='+')
plt.xlabel('Frame')
plt.ylabel('Cluster')
plt.title('iMWK-means with Explicit Rescaling and Kmeans')
plt.savefig(UserInput.out_name + '/RescalediMWK_timeseries.png')
plt.clf()
