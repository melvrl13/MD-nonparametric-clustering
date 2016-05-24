import numpy as np
import mdtraj as md
import sklearn.cluster
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib
matplotlib.use('Agg') #For use on DEAC cluster
import matplotlib.pyplot as plt
plt.style.use('bmh')
import hdbscan
import argparse

# Currently only working in python 2 due to MDAnalysis package.
# Initialize parser. The default help has poor labeling. See http://bugs.python.org/issue9694
parser = argparse.ArgumentParser(description = 'Run and score hdbscan clustering', add_help=False) 

# List all possible user input
inputs=parser.add_argument_group('Input arguments')
inputs.add_argument('-h', '--help', action='help')
inputs.add_argument('-top', action='store', dest='structure',help='Structure file corresponding to trajectory',type=str,required=True)
inputs.add_argument('-traj', action='store', dest='trajectory',help='Trajectory',type=str,required=True)
inputs.add_argument('-sel', action='store', dest='sel', help='Atom selection',type=str,default='not element H')
inputs.add_argument('-min', action='store', dest='min', help='minimum cluster membership',type=int,default=10)
inputs.add_argument('-o', action='store', dest='out_name',help='Output file',type=str,required=True)

# Parse into useful form
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
data = temp.reshape((frames,atoms*3))
data = data.astype('float64')
temp = []

# Run hdbscan
print("HDBSCAN trial")
clusterer = hdbscan.HDBSCAN(min_cluster_size=UserInput.min)
cluster_labels = clusterer.fit_predict(data)
if frames > 10000:
    sample_size = 10000
else:
    sample_size = None
raw_score = silhouette_score(data,cluster_labels, sample_size=sample_size)

# Save results
#Text
np.savetxt(UserInput.out_name + '/hdbscan_labels.txt', cluster_labels, fmt='%i')
with open (UserInput.out_name + '/silhouette_score.txt', 'w') as f:
    f.write("silhouette score is {0} \n".format(raw_score))
    
#Figures
plt.figure()
plt.scatter(np.arange(frames), cluster_labels, marker = '+')
plt.xlabel('Frame')
plt.ylabel('Cluster')
plt.title('HDBSCAN')
plt.savefig(UserInput.out_name + '/hdbscan_timeseries.png')
plt.clf()
