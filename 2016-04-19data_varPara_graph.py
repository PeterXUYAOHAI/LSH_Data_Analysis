from __future__ import division
print(__doc__)
import time
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import LSHForest
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn import preprocessing

from numpy import genfromtxt
my_data = genfromtxt('C:\Users\yaoha_000\Desktop\LSH_Project\\testoutput_small.csv', delimiter=',')

X = my_data[:,1:]
y = my_data[:,0]

np.random.shuffle(X)
X_norm = preprocessing.normalize(X, norm='l1',axis = 0)
X_index = X_norm[100:]
X_query = X_norm[:100]
rng = np.random.RandomState(42)

average_times_exact = []
average_times_approx = []
std_times_approx = []
accuracies = []
std_accuracies = []
average_speedups = []
std_speedups = []
n_samples=len(X_index)
outputString = ""

n_estimators_value = np.arange(0, 100, 20)
n_candidates_value = np.arange(10, 500, 100)
n_estimators_value[0]=1

for nestimators in n_estimators_value:
    for ncandidates in n_candidates_value:
        lshf = LSHForest(n_estimators=nestimators, n_candidates=ncandidates,
                     n_neighbors=10).fit(X_index)
        nbrs = NearestNeighbors(algorithm='brute', metric='cosine',
                            n_neighbors=10).fit(X_index)
        time_approx = []
        time_exact = []
        accuracy = []

        for i in range(5):
            query = X_query[[rng.randint(0, 100)]]

            t0 = time.time()
            exact_neighbors = nbrs.kneighbors(query, return_distance=False)
            time_exact.append(time.time() - t0)

            t0 = time.time()
            approx_neighbors = lshf.kneighbors(query, return_distance=False)
            time_approx.append(time.time() - t0)

            accuracy.append(np.in1d(approx_neighbors, exact_neighbors).mean())

        average_time_exact = np.mean(time_exact)
        average_time_approx = np.mean(time_approx)
        speedup = np.array(time_exact) / np.array(time_approx)
        average_speedup = np.mean(speedup)
        mean_accuracy = np.mean(accuracy)
        std_accuracy = np.std(accuracy)
        outputString += ("Index size: %d, estimators: %d, candidates: %d, exact: %0.3fs, LSHF: %0.3fs, speedup: %0.1f, "
              "accuracy: %0.2f +/-%0.2f" %
          (n_samples, nestimators,ncandidates, average_time_exact, average_time_approx, average_speedup,
           mean_accuracy, std_accuracy))
        print("Index size: %d, estimators: %d, candidates: %d, exact: %0.3fs, LSHF: %0.3fs, speedup: %0.1f, "
              "accuracy: %0.2f +/-%0.2f" %
          (n_samples, nestimators,ncandidates, average_time_exact, average_time_approx, average_speedup,
           mean_accuracy, std_accuracy))

        accuracies.append(mean_accuracy)
        std_accuracies.append(std_accuracy)
        average_times_exact.append(average_time_exact)
        average_times_approx.append(average_time_approx)
        std_times_approx.append(np.std(time_approx))
        average_speedups.append(average_speedup)
        std_speedups.append(np.std(speedup))


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np


#speed v accuracy graph
spacPair = zip(average_speedups, accuracies)     
spacPair = sorted(spacPair)
x,y = zip(*spacPair)

plt.figure()
plt.errorbar(x, y, yerr=std_times_approx,
             fmt='o-', c='r', label='speed v accuracy')
plt.ylabel("accuracy")
plt.xlabel("speedUp")


#esti vs candi vs speed
n_c_v = n_candidates_value
n_e_v = n_estimators_value
n_c_v, n_e_v = np.meshgrid(n_c_v, n_e_v)
n_speedup_v = np.asarray(average_speedups).reshape(5,5)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(n_c_v, n_e_v,n_speedup_v, rstride=1, cstride=1, 
                       linewidth=0, color='b',antialiased=False)
ax.set_xlabel('candidates')
ax.set_ylabel('estimators')

#esti vs candi vs speed vs accuracy 
n_c_v = n_candidates_value
n_e_v = n_estimators_value
n_c_v, n_e_v = np.meshgrid(n_c_v, n_e_v)
n_speedup_v_norm = np.squeeze(preprocessing.normalize( np.asarray(average_speedups).reshape(-1, 1), norm='l1',axis = 0))
n_accuracies_v_norm =np.squeeze(preprocessing.normalize( np.asarray(accuracies).reshape(-1, 1), norm='l1',axis = 0))
n_speedup_v = np.asarray(n_speedup_v_norm).reshape(5,5)
n_accuracy_v = np.asarray(n_accuracies_v_norm).reshape(5,5)
fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(n_c_v, n_e_v,n_accuracy_v, rstride=1, cstride=1, 
                       linewidth=0, color='b',antialiased=False)
surf = ax.plot_surface(n_c_v, n_e_v, n_speedup_v, rstride=1, cstride=1, 
                       linewidth=0, color='r',antialiased=False)

ax.set_xlabel('candidates')
ax.set_ylabel('estimators')

plt.show()