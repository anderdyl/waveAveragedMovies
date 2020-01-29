

import argusIO
import time as T
import numpy as np
import matplotlib.pyplot as plt
import datetime
import xarray as xr


file = 'grayWamImageSpaceC2_1572548400_20seconds_3secondDT.nc'

data = xr.open_dataset(file)

gray = data['gray'].values



new_gray = gray.reshape(-1, gray.shape[-1]).T


from sklearn import preprocessing
scalerMorph = preprocessing.StandardScaler().fit(new_gray)
X = preprocessing.scale(new_gray)

#
# #from sklearn import joblib
# #jolib.dum

#X = scalerMorph.transform(alllines)


from sklearn.decomposition import PCA

skpca = PCA()
skpca.fit(X)

f, ax = plt.subplots(figsize=(5,5))
ax.plot(skpca.explained_variance_ratio_[0:10]*100)
ax.plot(skpca.explained_variance_ratio_[0:10]*100,'ro')
ax.set_title("% of variance explained", fontsize=14)
ax.grid()

PCs = skpca.transform(X)
EOFs = skpca.components_
variance = skpca.explained_variance_
n_components = np.shape(PCs)[0]
n_features = np.shape(EOFs)[1]
pred_mean = alllines.mean(axis=0)
pred_std = alllines.std(axis=0)
scalerPCs = PCs/np.sqrt(variance[0])

