from collections import Counter
import numpy as np

features = np.array([
    [1, 2],
    [2, 3],
    [-1, -2],
    [3, 4],
])

labels = [[1, 2, 3, 4]]

test_points = np.array([[0, 0]])

d = np.sqrt(np.sum((features - test_points) ** 2, axis = 1))
dd = [labels[0][x] for x in np.argsort(d)]
print(dd)
votes = Counter(dd[:3])
print(votes.most_common(2))


