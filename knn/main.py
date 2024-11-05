from sklearn.neighbors import KNeighborsClassifier
from knn import DilisKnn
import numpy as np

features = [
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8]
]
features_np = np.array(features)
targets = [0, 0, 1, 1]
targets_np = np.array(targets)
# targets = np.array([[0],
#                     [0],
#                     [1],
#                     [1]])
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(features_np, targets_np)
l = knn.predict([[1, 1], [5, 5], [10, 10]])
print(l)

dknn = DilisKnn(n_neighbors = 3)
dknn.fit(features, targets)
dl = dknn.predict([[1, 1], [5, 5], [10, 10]])
# dl = dknn.predict([[1, 1]])
print(dl)