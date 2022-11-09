
import matplotlib.pyplot as plt
import numpy as np
import gudhi as tda
import pandas as pd
from sklearn.decomposition import PCA
from gtda.time_series import SingleTakensEmbedding
from gtda.plotting import plot_point_cloud
from mpl_toolkits import mplot3d
import plotly.graph_objs as gobj

import scipy.io
df = scipy.io.loadmat('C:\Work\datasets\WFDBRecords\\01\\010\JS00001.mat')
df = df['val']
series = df[5]

embedding_dimension = 3
embedding_time_delay = 3
stride = 2

embedder = SingleTakensEmbedding(
    parameters_type="search", n_jobs=6, time_delay=embedding_time_delay, dimension=embedding_dimension, stride=stride
)


ECG_embedded = embedder.fit_transform(series)
pca = PCA(n_components=3)
y_axis = pca.fit_transform(ECG_embedded)
df = pd.DataFrame(ECG_embedded)
fig = plt.figure()
#ax = plt.axes(projection = '3d')

x = df.iloc[:,0]
y = df.iloc[:,1]
z = df.iloc[:,2]
#ax.scatter3D(x,y,z)
plot_point_cloud(ECG_embedded, dimension=3)
fig.show()

plt.show()


