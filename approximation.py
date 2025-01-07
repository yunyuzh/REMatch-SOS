import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from dscribe.kernels import REMatchKernel

def load_soap(directory):
    features_list = []
    indices = []
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

def breakdown(features_list, num_groups):
    groups = np.array_split(features_list, num_groups)
    return groups

def rematch_for_same_group(groups, matrix, re):
    for i in range(10):
        re_kernel = re.create(groups[i])
        start_index = i * 1000
        end_index = start_index + 1000
        matrix[start_index:end_index, start_index:end_index] = re_kernel
    return matrix

def rematch_for_cross_groups(groups, matrix, re):
    for i in range(9):
        re_kernel = re.create(groups[i],groups[i+1])
        start_index_i = i * 1000
        end_index_i = start_index_i + 1000
        start_index_j = (i + 1) * 1000
        end_index_j = start_index_j + 1000
        matrix[start_index_i:end_index_i, start_index_j:end_index_j] = re_kernel
        matrix[start_index_j:end_index_j, start_index_i:end_index_i] = re_kernel.T
    return matrix

def get_distance_matrix(matrix):
    distance_matrix = 1 - matrix
    distance_matrix = np.clip(distance_matrix, 0, None)
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix

def update_distance_matrix(matrix):

    for i in range(9):
        group_start_index = i * 1000
        group_end_index = group_start_index + 1000

        for j in range(1000):
            current_index = group_start_index + j

            chain = []

            if np.any(matrix[current_index, :] == 1):
                
                chain.append(current_index)

                for k in range(i + 1, 10):
                    next_group_start_index = k * 1000
                    next_group_end_index = next_group_start_index + 1000

                    last_index_in_chain = chain[-1]
                    distances_to_next_group = matrix[last_index_in_chain, next_group_start_index:next_group_end_index]
                    nearest_index_in_next_group = np.argmin(distances_to_next_group) + next_group_start_index
                    chain.append(nearest_index_in_next_group)

                for l in range(1, len(chain)-1):
                    current_structure_index = chain[l]
                    next_structure_index = chain[l + 1]
                    next_structure_group_start = (next_structure_index // 1000) * 1000
                    next_structure_group_end = next_structure_group_start + 1000

                    for m in range(l):
                        m_structure_index = chain[m]
                        matrix[m_structure_index, next_structure_group_start:next_structure_group_end] = matrix[current_structure_index, next_structure_group_start:next_structure_group_end]
                        matrix[next_structure_group_start:next_structure_group_end, m_structure_index] = matrix[next_structure_group_start:next_structure_group_end, current_structure_index]

    return matrix

def process_rematch(file_base_path, output_path):
    features, indices, gamma = load_soap(file_base_path)
    groups = breakdown(features, 10)
    matrix = np.zeros((10000,10000))
    re = REMatchKernel(metric="rbf", gamma=gamma, alpha=1, threshold=1e-6)
    matrix = rematch_for_same_group(groups, matrix, re)
    matrix = rematch_for_cross_groups(groups, matrix, re)
    distance_matrix = get_distance_matrix(matrix)
    distance_matrix = update_distance_matrix(distance_matrix)
    np.save(output_path, distance_matrix)
