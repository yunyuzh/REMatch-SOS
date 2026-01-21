import os
import numpy as np
from sklearn.preprocessing import normalize
from dscribe.kernels import REMatchKernel

def load_soap(directory):
    features_list = []
    indices = []

    # Change 10000 to however many structures you have (A0.csv ... A{N-1}.csv)
    for i in range(10000):
        file_path = os.path.join(directory, f'A{i}.csv')
        if os.path.exists(file_path):
            features = np.loadtxt(file_path, delimiter=',')
            features = normalize(features)
            features_list.append(features)
            indices.append(i)
    features = np.vstack(features_list)
    variance = np.var(features)
    n_features = features.shape[1]
    gamma = 1 / (n_features * variance)
    return features_list, indices, gamma

def process_rematch(file_base_path, output_path):
    features, indices, gamma = load_soap(file_base_path)
    re = REMatchKernel(metric="rbf", gamma=gamma, alpha=1, threshold=1e-6)
    matrix = re.create(features)
    distance_matrix = get_distance_matrix(matrix)
    np.save(output_path, distance_matrix)

def get_distance_matrix(matrix):
    distance_matrix = 1 - matrix
    distance_matrix = np.clip(distance_matrix, 0, None)
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix
