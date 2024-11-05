from collections import Counter
from math import sqrt
import numpy as np

class DilisKnn:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        self.observations = None
        self.labels = None

    def fit(self, features, targets):
        self.observations = features
        self.labels = targets

    def label_for_point(self, point):
        '''
        Non-Numpy
        '''
        distances = []
        # print(f'point: {point}')
        for obs_id, observation in enumerate(self.observations):
            # print(f'obs: {observation}')
            # for f in range(len(observation)):
            #     print(f'  {f}: {point[f]}, {observation[f]} = {(point[f] - observation[f]) ** 2}')
            distances.append(
                [self.labels[obs_id], sqrt(sum([(point[f] - observation[f]) ** 2 for f in range(len(observation))]))]
            )
        sorted_distances = sorted(distances, key=lambda x: x[1])
        # print(distances)
        # print(sorted_distances)
        return sorted_distances[0][0]
    
    def label_for_point_np(self, point):
        '''
        Using numpy
        '''
        distances = np.sqrt(np.sum((self.observations - point) ** 2, axis=1))
        sorted_ids = [self.labels[idx] for idx in np.argsort(distances)]
        votes = Counter(sorted_ids[:self.n_neighbors])
        return votes.most_common(1)[0][0]

    def predict(self, points):
        # return list([self.label_for_point(point) for point in points])
        return [self.label_for_point(point) for point in points]
        # return [0, 0, 0]
            

